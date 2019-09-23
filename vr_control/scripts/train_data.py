#!/usr/bin/env python

import numpy as np
import random
import tensorflow as tf
import imageio

from tensorflow.python.platform import flags
from network.mil_network import MIL
from network.prepare_data import *

FLAGS = flags.FLAGS

# Directories
flags.DEFINE_integer('dataset_number', None, 'Number of dataset')
flags.DEFINE_string('output', None, 'vel or dmp')
flags.DEFINE_string('experiment', 'sim_' + FLAGS.output + '_' + str(FLAGS.dataset_number), 'name of the experiment')
flags.DEFINE_string('demo_file', '/home/mtheofanidis/catkin_ws/src/Data/Unreal-Dataset/Dataset_' + str(FLAGS.dataset_number), 'states and actions')
flags.DEFINE_string('demo_gif_dir', '/home/mtheofanidis/catkin_ws/src/Data/Unreal-Dataset/Dataset_' + str(FLAGS.dataset_number), 'videos')
flags.DEFINE_string('scale_dir', '/home/mtheofanidis/catkin_ws/src/Data/Unreal-Dataset/scale_and_bias_%s.pkl' % FLAGS.experiment, 'storage')
flags.DEFINE_string('temp_dir', None, 'tempory directory for testing')
flags.DEFINE_string('gif_prefix', 'object', 'prefix of the video directory for each task, e.g. object_0 for task 0')
flags.DEFINE_integer('restore_iter', 0, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('hsv', False, 'convert the image to HSV format')
flags.DEFINE_bool('use_noisy_demos', False, 'use noisy demonstrations or not (for domain shift)')

# Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('log_dir', '/home/mtheofanidis/catkin_ws/src/network/tmp/data', 'summaries and checkpoints.')
flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_update_batch_size', 1, 'number of demos used during test time')
flags.DEFINE_float('gpu_memory_fraction', 0.5, 'fraction of memory used in gpu')
flags.DEFINE_bool('record_gifs', True, 'record gifs during evaluation')

# Training Options
flags.DEFINE_integer('meta_batch_size', 1, 'number of tasks sampled per meta-update')
flags.DEFINE_integer('val_set_size', 1, 'size of the training set')
flags.DEFINE_integer('T', 100, 'time horizon of the demo videos')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training .')
flags.DEFINE_integer('im_width', 125, 'width of the images in the demo videos')
flags.DEFINE_integer('im_height', 125, 'height of the images in the demo videos')
flags.DEFINE_integer('num_channels', 3, 'number of channels of the images in the demo videos')
flags.DEFINE_integer('metatrain_iterations', 30000, 'number of metatraining iterations.')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 1, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('train_update_lr', 0.01, 'step size alpha for inner gradient update.')
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')
flags.DEFINE_bool('clip', False, 'use gradient clipping for fast gradient')
flags.DEFINE_float('clip_max', 10.0, 'maximum clipping value for fast gradient')
flags.DEFINE_float('clip_min', -10.0, 'minimum clipping value for fast gradient')
flags.DEFINE_bool('fc_bt', True, 'use bias transformation for the first fc layer')
flags.DEFINE_bool('all_fc_bt', False, 'use bias transformation for all fc layers')
flags.DEFINE_bool('conv_bt', True, 'use bias transformation for the first conv layer, N/A for using pretraining')
flags.DEFINE_integer('bt_dim', 20, 'the dimension of bias transformation for FC layers')
flags.DEFINE_string('pretrain_weight_path', 'N/A', 'path to pretrained weights')
flags.DEFINE_bool('train_pretrain_conv1', False, 'whether to finetune the pretrained weights')
flags.DEFINE_bool('two_head', True, 'use two-head architecture')
flags.DEFINE_bool('learn_final_eept', False, 'learn an auxiliary loss for predicting final end-effector pose')
flags.DEFINE_bool('learn_final_eept_whole_traj', False, 'learn an auxiliary loss for predicting final end-effector pose by passing the whole trajectory of eepts (used for video-only models)')
flags.DEFINE_bool('stopgrad_final_eept', True, 'stop the gradient when concatenate the predicted final eept with the feature points')
flags.DEFINE_integer('final_eept_min', 6, 'first index of the final eept in the action array')
flags.DEFINE_integer('final_eept_max', 8, 'last index of the final eept in the action array')
flags.DEFINE_float('final_eept_loss_eps', 0.1, 'the coefficient of the auxiliary loss')
flags.DEFINE_float('act_loss_eps', 1.0, 'the coefficient of the action loss')
flags.DEFINE_float('loss_multiplier', 50.0, 'the constant multiplied with the loss value, 100 for reach and 50 for push')
flags.DEFINE_bool('use_l1_l2_loss', False, 'use a loss with combination of l1 and l2')
flags.DEFINE_float('l2_eps', 0.01, 'coeffcient of l2 loss')
flags.DEFINE_bool('shuffle_val', False, 'whether to choose the validation set via shuffling or not')
flags.DEFINE_bool('no_action', False, 'do not include actions in the demonstrations for inner update')
flags.DEFINE_bool('no_state', False, 'do not include states in the demonstrations during training')
flags.DEFINE_bool('no_final_eept', False, 'do not include final ee pos in the demonstrations for inner update')
flags.DEFINE_bool('zero_state', False, 'zero-out states (meta-learn state) in the demonstrations for inner update (used in the paper with video-only demos)')
flags.DEFINE_bool('two_arms', False, 'use two-arm structure when state is zeroed-out')
flags.DEFINE_integer('training_set_size', -1, 'size of the training set, -1 for all data except those in validation set')

# Model Options
flags.DEFINE_integer('random_seed', 0, 'random seed for training')
flags.DEFINE_bool('fp', True, 'use spatial soft-argmax or not')
flags.DEFINE_string('norm', 'layer_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_bool('dropout', False, 'use dropout for fc layers or not')
flags.DEFINE_float('keep_prob', 0.5, 'keep probability for dropout')
flags.DEFINE_integer('num_filters', 16, 'number of filters for conv nets')
flags.DEFINE_integer('filter_size', 5, 'filter size for conv nets')
flags.DEFINE_integer('num_conv_layers', 4, 'number of conv layers')
flags.DEFINE_integer('num_strides', 4, 'number of conv layers with strided filters')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_integer('num_fc_layers', 3, 'number of fully-connected layers')
flags.DEFINE_integer('layer_size', 200, 'hidden dimension of fully-connected layers')
flags.DEFINE_bool('temporal_conv_2_head', False, 'whether or not to use temporal convolutions for the two-head architecture in video-only setting.')
flags.DEFINE_bool('temporal_conv_2_head_ee', False, 'whether or not to use temporal convolutions for the two-head architecture in video-only setting for predicting the ee pose.')
flags.DEFINE_integer('temporal_filter_size', 5, 'filter size for temporal convolution')
flags.DEFINE_integer('temporal_num_filters', 64, 'number of filters for temporal convolution')
flags.DEFINE_integer('temporal_num_filters_ee', 64, 'number of filters for temporal convolution for ee pose prediction')
flags.DEFINE_integer('temporal_num_layers', 3, 'number of layers for temporal convolution for ee pose prediction')
flags.DEFINE_integer('temporal_num_layers_ee', 3, 'number of layers for temporal convolution for ee pose prediction')
flags.DEFINE_string('init', 'xavier', 'initializer for conv weights. Choose among random, xavier, and he')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')


def train(graph, model, saver, sess, data_generator, log_dir, restore_itr=0):

    PRINT_INTERVAL = 100
    TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 1000
    TOTAL_ITERS = FLAGS.metatrain_iterations
    prelosses, postlosses = [], []
    save_dir = log_dir + '/model'

    train_writer = tf.summary.FileWriter(log_dir, graph)

    # actual training.
    if restore_itr == 0:
        training_range = range(TOTAL_ITERS)
    else:
        training_range = range(restore_itr+1, TOTAL_ITERS)
    for itr in training_range:
        state, tgt_mu = data_generator.generate_data_batch(itr)
        statea = state[:, :FLAGS.update_batch_size*FLAGS.T, :]
        stateb = state[:, FLAGS.update_batch_size*FLAGS.T:, :]
        actiona = tgt_mu[:, :FLAGS.update_batch_size*FLAGS.T, :]
        actionb = tgt_mu[:, FLAGS.update_batch_size*FLAGS.T:, :]
        feed_dict = {model.statea: statea,
                    model.stateb: stateb,
                    model.actiona: actiona,
                    model.actionb: actionb}
        input_tensors = [model.train_op]

        if itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0:
            input_tensors.extend([model.train_summ_op, model.total_loss1, model.total_losses2[model.num_updates-1]])
        with graph.as_default():
            results = sess.run(input_tensors, feed_dict=feed_dict)

        if itr != 0 and itr % SUMMARY_INTERVAL == 0:
            prelosses.append(results[-2])
            train_writer.add_summary(results[-3], itr)
            postlosses.append(results[-1])

        if itr != 0 and itr % PRINT_INTERVAL == 0:
            print 'Iteration %d: average preloss is %.2f, average postloss is %.2f' % (itr, np.mean(prelosses), np.mean(postlosses))
            prelosses, postlosses = [], []

        if itr != 0 and itr % TEST_PRINT_INTERVAL == 0:
            if FLAGS.val_set_size > 0:
                input_tensors = [model.val_summ_op, model.val_total_loss1, model.val_total_losses2[model.num_updates-1]]
                val_state, val_act = data_generator.generate_data_batch(itr, train=False)
                statea = val_state[:, :FLAGS.update_batch_size*FLAGS.T, :]
                stateb = val_state[:, FLAGS.update_batch_size*FLAGS.T:, :]
                actiona = val_act[:, :FLAGS.update_batch_size*FLAGS.T, :]
                actionb = val_act[:, FLAGS.update_batch_size*FLAGS.T:, :]
                feed_dict = {model.statea: statea,
                            model.stateb: stateb,
                            model.actiona: actiona,
                            model.actionb: actionb}
                with graph.as_default():
                    results = sess.run(input_tensors, feed_dict=feed_dict)
                train_writer.add_summary(results[0], itr)
                print 'Test results: average preloss is %.2f, average postloss is %.2f' % (np.mean(results[1]), np.mean(results[2]))

        if itr != 0 and (itr % SAVE_INTERVAL == 0 or itr == training_range[-1]):
            print 'Saving model to: %s' % (save_dir + '_%d' % itr)
            with graph.as_default():
                saver.save(sess, save_dir + '_%d' % itr)


def generate_test_demos(data_generator):
    if not FLAGS.use_noisy_demos:
        n_folders = len(data_generator.demos.keys())
        demos = data_generator.demos
    else:
        n_folders = len(data_generator.noisy_demos.keys())
        demos = data_generator.noisy_demos
    policy_demo_idx = [np.random.choice(n_demo, replace=False, size=FLAGS.test_update_batch_size) \
                        for n_demo in [demos[i]['demoX'].shape[0] for i in xrange(n_folders)]]
    selected_demoO, selected_demoX, selected_demoU = [], [], []
    for i in xrange(n_folders):
        selected_cond = np.array(demos[i]['demoConditions'])[np.arange(len(demos[i]['demoConditions'])) == policy_demo_idx[i]]
        Xs, Us, Os = [], [], []
        for idx in selected_cond:
            if FLAGS.use_noisy_demos:
                demo_gif_dir = data_generator.noisy_demo_gif_dir
            else:
                demo_gif_dir = data_generator.demo_gif_dir
            O = np.array(imageio.mimread(demo_gif_dir + data_generator.gif_prefix + '_%d/cond%d.samp0.gif' % (i, idx)))[:, :, :, :3]
            O = np.transpose(O, [0, 3, 2, 1]) # transpose to mujoco setting for images
            O = O.reshape(FLAGS.T, -1) / 255.0 # normalize
            Os.append(O)
        Xs.append(demos[i]['demoX'][np.arange(demos[i]['demoX'].shape[0]) == policy_demo_idx[i]].squeeze())
        Us.append(demos[i]['demoU'][np.arange(demos[i]['demoU'].shape[0]) == policy_demo_idx[i]].squeeze())
        selected_demoO.append(np.array(Os))
        selected_demoX.append(np.array(Xs))
        selected_demoU.append(np.array(Us))
    print "Finished collecting demos for testing"
    selected_demo = dict(selected_demoX=selected_demoX, selected_demoU=selected_demoU, selected_demoO=selected_demoO)
    data_generator.selected_demo = selected_demo


def main():

    tf.set_random_seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)
    random.seed(FLAGS.random_seed)

    graph = tf.Graph()

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    sess = tf.Session(graph=graph, config=tf_config)

    network_config = {
        'num_filters': [FLAGS.num_filters]*FLAGS.num_conv_layers,
        'strides': [[1, 2, 2, 1]]*FLAGS.num_strides + [[1, 1, 1, 1]]*(FLAGS.num_conv_layers-FLAGS.num_strides),
        'filter_size': FLAGS.filter_size,
        'image_width': FLAGS.im_width,
        'image_height': FLAGS.im_height,
        'image_channels': FLAGS.num_channels,
        'n_layers': FLAGS.num_fc_layers,
        'layer_size': FLAGS.layer_size,
        'initialization': FLAGS.init,
    }
    data_generator = DataGenerator()
    state_idx = data_generator.state_idx
    img_idx = range(len(state_idx), len(state_idx)+FLAGS.im_height*FLAGS.im_width*FLAGS.num_channels)

    # need to compute x_idx and img_idx from data_generator
    model = MIL(data_generator._dU, state_idx=state_idx, img_idx=img_idx, network_config=network_config)

    # TODO: figure out how to save summaries and checkpoints
    exp_string = FLAGS.experiment+ '.' + FLAGS.init + '_init.' + str(FLAGS.num_conv_layers) + '_conv' + '.' + str(FLAGS.num_strides) + '_strides' + '.' + str(FLAGS.num_filters) + '_filters' + \
                '.' + str(FLAGS.num_fc_layers) + '_fc' + '.' + str(FLAGS.layer_size) + '_dim' + '.bt_dim_' + str(FLAGS.bt_dim) + '.mbs_'+str(FLAGS.meta_batch_size) + \
                '.ubs_' + str(FLAGS.update_batch_size) + '.numstep_' + str(FLAGS.num_updates) + '.updatelr_' + str(FLAGS.train_update_lr)

    if FLAGS.clip:
        exp_string += '.clip_' + str(int(FLAGS.clip_max))
    if FLAGS.conv_bt:
        exp_string += '.conv_bt'
    if FLAGS.all_fc_bt:
        exp_string += '.all_fc_bt'
    if FLAGS.fp:
        exp_string += '.fp'
    if FLAGS.learn_final_eept:
        exp_string += '.learn_ee_pos'
    if FLAGS.no_action:
        exp_string += '.no_action'
    if FLAGS.zero_state:
        exp_string += '.zero_state'
    if FLAGS.two_head:
        exp_string += '.two_heads'
    if FLAGS.two_arms:
        exp_string += '.two_arms'
    if FLAGS.temporal_conv_2_head:
        exp_string += '.1d_conv_act_' + str(FLAGS.temporal_num_layers) + '_' + str(FLAGS.temporal_num_filters)
        if FLAGS.temporal_conv_2_head_ee:
            exp_string += '_ee_' + str(FLAGS.temporal_num_layers_ee) + '_' + str(FLAGS.temporal_num_filters_ee)
        exp_string += '_' + str(FLAGS.temporal_filter_size) + 'x1_filters'
    if FLAGS.training_set_size != -1:
        exp_string += '.' + str(FLAGS.training_set_size) + '_trials'

    log_dir = FLAGS.log_dir + '/' + exp_string

    # put here for now
    if FLAGS.train:
        data_generator.generate_batches()
        with graph.as_default():
            train_image_tensors = data_generator.make_batch_tensor(network_config, restore_iter=FLAGS.restore_iter)
            inputa = train_image_tensors[:, :FLAGS.update_batch_size*FLAGS.T, :]
            inputb = train_image_tensors[:, FLAGS.update_batch_size*FLAGS.T:, :]
            train_input_tensors = {'inputa': inputa, 'inputb': inputb}
            val_image_tensors = data_generator.make_batch_tensor(network_config, restore_iter=FLAGS.restore_iter, train=False)
            inputa = val_image_tensors[:, :FLAGS.update_batch_size*FLAGS.T, :]
            inputb = val_image_tensors[:, FLAGS.update_batch_size*FLAGS.T:, :]
            val_input_tensors = {'inputa': inputa, 'inputb': inputb}
        model.init_network(graph, input_tensors=train_input_tensors, restore_iter=FLAGS.restore_iter)
        model.init_network(graph, input_tensors=val_input_tensors, restore_iter=FLAGS.restore_iter, prefix='Validation_')
    else:
        model.init_network(graph, prefix='Testing')
    with graph.as_default():
        # Set up saver.
        saver = tf.train.Saver(max_to_keep=10)
        # Initialize variables.
        init_op = tf.global_variables_initializer()
        sess.run(init_op, feed_dict=None)
        # Start queue runners (used for loading videos on the fly)
        tf.train.start_queue_runners(sess=sess)
    if FLAGS.resume:

        model_file = tf.train.latest_checkpoint(FLAGS.temp_dir)
        if FLAGS.restore_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model_' + str(FLAGS.restore_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+6:])
            print("Restoring model weights from " + model_file)
            with graph.as_default():
                saver.restore(sess, model_file)
    if FLAGS.train:
        print("Train")
        train(graph, model, saver, sess, data_generator, log_dir, restore_itr=FLAGS.restore_iter)


if __name__ == "__main__":
    main()

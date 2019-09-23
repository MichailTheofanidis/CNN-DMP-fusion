#!/usr/bin/env python

import glob
import pickle
import random
import numpy as np
import tensorflow as tf

from collections import OrderedDict
from utils import *
from tensorflow.python.platform import flags
from natsort import natsorted

FLAGS = flags.FLAGS


class DataGenerator(object):

    def __init__(self):

        # Hyperparameters
        self.update_batch_size = FLAGS.update_batch_size
        self.test_batch_size = FLAGS.train_update_batch_size if FLAGS.train_update_batch_size != -1 else self.update_batch_size
        self.meta_batch_size = FLAGS.meta_batch_size
        self.T = FLAGS.T
        self.demo_gif_dir = FLAGS.demo_gif_dir
        self.gif_prefix = FLAGS.gif_prefix
        self.restore_iter = FLAGS.restore_iter
        # Scale and bias for data normalization
        self.scale, self.bias = None, None

        demo_file = FLAGS.demo_file
        demo_file = natsorted(glob.glob(demo_file + '/*pkl'))
        self.dataset_size = len(demo_file)
        if FLAGS.train and FLAGS.training_set_size != -1:
            tmp = demo_file[:FLAGS.training_set_size]
            tmp.extend(demo_file[-FLAGS.val_set_size:])
            demo_file = tmp
        self.extract_supervised_data(demo_file)

    def unpickle(self, filename):

        result = pickle.load(filename)

        return result

    def extract_demo_dict(self, demo_file):
        demos = {}
        for i in range(0, len(demo_file)):
            demos[i] = self.unpickle(open(demo_file[i], 'rb'))

        return demos

    def extract_supervised_data(self, demo_file, noisy=False):

        demos = self.extract_demo_dict(demo_file)
        n_folders = len(demos.keys())
        N_demos = np.sum(demo['demoX'].shape[0] for i, demo in demos.iteritems())
        self.state_idx = range(demos[0]['demoX'].shape[-1])

        if FLAGS.output == 'vel':
            self._dU = demos[0]['demoU'].shape[-1]
        elif FLAGS.output == 'dmp':
            self._dU = demos[0]['demoDMP'].shape[2]

        print "Number of demos: %d" % N_demos
        idx = np.arange(n_folders)
        if FLAGS.train:
            n_val = FLAGS.val_set_size  # number of demos for testing
            if not hasattr(self, 'train_idx'):
                if n_val != 0:
                    if not FLAGS.shuffle_val:
                        self.val_idx = idx[-n_val:]
                        self.train_idx = idx[:-n_val]
                    else:
                        self.val_idx = np.sort(np.random.choice(idx, size=n_val, replace=False))
                        mask = np.array([(i in self.val_idx) for i in idx])
                        self.train_idx = np.sort(idx[~mask])
                else:
                    self.train_idx = idx
                    self.val_idx = []
            # Normalize the states if it's training.
            if self.scale is None or self.bias is None:
                states = np.vstack((demos[i]['demoX'] for i in self.train_idx))  # hardcoded here to solve the memory issue
                states = states.reshape(-1, len(self.state_idx))

                # 1e-3 to avoid infs if some state dimensions don't change in the
                # first batch of samples
                self.scale = np.diag(1.0 / np.maximum(np.std(states, axis=0), 1e-3))
                self.bias = - np.mean(states.dot(self.scale), axis=0)

                # Save the scale and bias.
                with open(FLAGS.scale_dir, 'wb') as f:
                    pickle.dump({'scale': self.scale, 'bias': self.bias}, f)
            for key in demos.keys():
                demos[key]['demoX'] = demos[key]['demoX'].reshape(-1, len(self.state_idx))
                demos[key]['demoX'] = demos[key]['demoX'].dot(self.scale) + self.bias
                demos[key]['demoX'] = demos[key]['demoX'].reshape(-1, self.T, len(self.state_idx))

            self.demos = demos

    def generate_batches(self, noisy=False):
        with Timer('Generating batches for each iteration'):
            if FLAGS.training_set_size != -1:
                offset = self.dataset_size - FLAGS.training_set_size - FLAGS.val_set_size
            else:
                offset = 0
            img_folders = natsorted(glob.glob(FLAGS.demo_file + '/' + self.gif_prefix + '_*'))

            train_img_folders = {i: img_folders[i] for i in self.train_idx}
            val_img_folders = {i: img_folders[i + offset] for i in self.val_idx}

            TEST_PRINT_INTERVAL = 500
            TOTAL_ITERS = FLAGS.metatrain_iterations
            self.all_training_filenames = []
            self.all_val_filenames = []
            self.training_batch_idx = {i: OrderedDict() for i in xrange(TOTAL_ITERS)}
            self.val_batch_idx = {i: OrderedDict() for i in
                                  TEST_PRINT_INTERVAL * np.arange(1, int(TOTAL_ITERS / TEST_PRINT_INTERVAL))}

            for itr in xrange(TOTAL_ITERS):

                sampled_train_idx = random.sample(self.train_idx, self.meta_batch_size)
                for idx in sampled_train_idx:
                    sampled_folder = train_img_folders[idx]
                    image_paths = natsorted(os.listdir(sampled_folder))

                    try:
                        assert len(image_paths) == self.demos[idx]['demoX'].shape[0]
                    except AssertionError:
                        import pdb;
                        pdb.set_trace()

                    sampled_image_idx = np.random.choice(range(len(image_paths)), size=self.update_batch_size + self.test_batch_size, replace=True)  # True
                    sampled_images = [os.path.join(sampled_folder, image_paths[i]) for i in sampled_image_idx]

                    self.all_training_filenames.extend(sampled_images)
                    self.training_batch_idx[itr][idx] = sampled_image_idx

                if itr != 0 and itr % TEST_PRINT_INTERVAL == 0:
                    sampled_val_idx = random.sample(self.val_idx, self.meta_batch_size)
                    for idx in sampled_val_idx:
                        sampled_folder = val_img_folders[idx]
                        image_paths = natsorted(os.listdir(sampled_folder))
                        assert len(image_paths) == self.demos[idx]['demoX'].shape[0]

                        sampled_image_idx = np.random.choice(range(len(image_paths)),
                                                                 size=self.update_batch_size + self.test_batch_size,
                                                                 replace=True)  # True
                        sampled_images = [os.path.join(sampled_folder, image_paths[i]) for i in sampled_image_idx]
                        self.all_val_filenames.extend(sampled_images)
                        self.val_batch_idx[itr][idx] = sampled_image_idx


    def make_batch_tensor(self, network_config, restore_iter=0, train=True):
        TEST_INTERVAL = 500
        batch_image_size = (self.update_batch_size + self.test_batch_size) * self.meta_batch_size
        if train:
            all_filenames = self.all_training_filenames
            if restore_iter > 0:
                all_filenames = all_filenames[batch_image_size * (restore_iter + 1):]
        else:
            all_filenames = self.all_val_filenames
            if restore_iter > 0:
                all_filenames = all_filenames[batch_image_size * (int(restore_iter / TEST_INTERVAL) + 1):]
        im_height = network_config['image_height']
        im_width = network_config['image_width']
        num_channels = network_config['image_channels']
        # make queue for tensorflow to read from
        filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
        print 'Generating image processing ops'
        image_reader = tf.WholeFileReader()
        _, image_file = image_reader.read(filename_queue)
        image = tf.image.decode_gif(image_file)
        # should be T x C x W x H
        image.set_shape((self.T, im_height, im_width, num_channels))
        image = tf.cast(image, tf.float32)
        image /= 255.0
        if FLAGS.hsv:
            eps_min, eps_max = 0.5, 1.5
            assert eps_max >= eps_min >= 0
            # convert to HSV only fine if input images in [0, 1]
            img_hsv = tf.image.rgb_to_hsv(image)
            img_h = img_hsv[..., 0]
            img_s = img_hsv[..., 1]
            img_v = img_hsv[..., 2]
            eps = tf.random_uniform([self.T, 1, 1], eps_min, eps_max)
            img_v = tf.clip_by_value(eps * img_v, 0., 1.)
            img_hsv = tf.stack([img_h, img_s, img_v], 3)
            image_rgb = tf.image.hsv_to_rgb(img_hsv)
            image = image_rgb
        image = tf.transpose(image, perm=[0, 3, 2, 1])  # transpose to mujoco setting for images
        image = tf.reshape(image, [self.T, -1])
        num_preprocess_threads = 1  # TODO - enable this to be set to >1
        min_queue_examples = 64  # 128 #256
        print 'Batching images'
        images = tf.train.batch(
            [image],
            batch_size=batch_image_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_image_size,
        )
        all_images = []
        for i in xrange(self.meta_batch_size):
            image = images[i * (self.update_batch_size + self.test_batch_size):(i + 1) * (
                        self.update_batch_size + self.test_batch_size)]
            image = tf.reshape(image, [(self.update_batch_size + self.test_batch_size) * self.T, -1])
            all_images.append(image)
        return tf.stack(all_images)

    def generate_data_batch(self, itr, train=True):
        if train:
            demos = {key: self.demos[key].copy() for key in self.train_idx}
            idxes = self.training_batch_idx[itr]

        else:
            demos = {key: self.demos[key].copy() for key in self.val_idx}
            idxes = self.val_batch_idx[itr]

        batch_size = self.meta_batch_size
        update_batch_size = self.update_batch_size
        test_batch_size = self.test_batch_size

        if FLAGS.output == 'vel':

            U = [demos[k]['demoU'][v].reshape((test_batch_size + update_batch_size) * self.T, -1) for k, v in idxes.items()]
            U = np.array(U)
            X = [demos[k]['demoX'][v].reshape((test_batch_size + update_batch_size) * self.T, -1) for k, v in idxes.items()]
            X = np.array(X)

        elif FLAGS.output == 'dmp':

            U = [demos[k]['demoDMP'][v].reshape((test_batch_size + update_batch_size) * self.T, -1) for k, v in idxes.items()]
            U = np.array(U)
            X = [demos[k]['demoX'][v].reshape((test_batch_size + update_batch_size) * self.T, -1) for k, v in idxes.items()]
            X = np.array(X)

        assert U.shape[2] == self._dU
        assert X.shape[2] == len(self.state_idx)
        return X, U


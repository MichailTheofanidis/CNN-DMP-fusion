#!/usr/bin/env python

import cv2
import math
import random
import imageio
import numpy as np
import tensorflow as tf

from robot_class.recorder import Recorder
from robot_class.dmp import DynamicMovementPrimitive as DMP
from tensorflow.python.platform import flags
from robot_class.robot import MyRobot as Robot
from network.mil_network import MIL
from network.prepare_data import *
from robot_class.environment import *
from robot_class.utils import *
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from service_pkg.srv import *

FLAGS = flags.FLAGS

FINAL_THRESHOLD = 0.22
FINAL_THRESHOLD_T = 0.05
FAIL = 1

# Directories
flags.DEFINE_integer('dataset_number', 3, 'Number of dataset')
flags.DEFINE_string('output', 'vel', 'vel or dmp')
flags.DEFINE_string('experiment', 'sim_' + FLAGS.output + '_' + str(FLAGS.dataset_number), 'name of the experiment')
flags.DEFINE_string('demo_file', '/home/mtheofanidis/catkin_ws/src/Data/Unreal-Dataset/Dataset_' + str(FLAGS.dataset_number), 'states and actions')
flags.DEFINE_string('demo_gif_dir', '/home/mtheofanidis/catkin_ws/src/Data/Unreal-Dataset/Dataset_' + str(FLAGS.dataset_number), 'videos')
flags.DEFINE_string('scale_dir', '/home/mtheofanidis/catkin_ws/src/Data/Unreal-Dataset/scale_and_bias_%s.pkl' % FLAGS.experiment, 'storage')
flags.DEFINE_string('test_dir', '/home/mtheofanidis/catkin_ws/src/Data/Unreal-Dataset/Dataset_1', 'testing demos')
flags.DEFINE_string('temp_dir', None, 'tempory directory for testing')
flags.DEFINE_string('gif_prefix', 'object', 'prefix of the video directory for each task, e.g. object_0 for task 0')
flags.DEFINE_integer('restore_iter', 0, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('hsv', False, 'convert the image to HSV format')
flags.DEFINE_bool('use_noisy_demos', False, 'use noisy demonstrations or not (for domain shift)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('log_dir', '/home/mtheofanidis/catkin_ws/src/network/tmp/data', 'summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', False, 'True to train, False to test.')
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


def load_scale_and_bias(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        scale = data['scale']
        bias = data['bias']
    return scale, bias


def load_demo(task_id, demo_dir, demo_inds):

    file = natsorted(glob.glob(demo_dir + '/*pkl'))
    demo_info = pickle.load(open(file[1], 'rb'))

    if FLAGS.output is 'vel':
        demoX = demo_info['demoX']
        demoU = demo_info['demoU']

    elif FLAGS.output is 'dmp':
        demoX = demo_info['demoX']
        demoU = demo_info['demoDMP']

    demo_gifs = imageio.mimread(demo_dir+'/object_'+task_id+'/demo_%d.gif' % demo_inds)

    return demoX, demoU, demo_gifs, demo_info


def get_snapshot():

    for i in range(0, 10):

        s = rospy.ServiceProxy("return_camera_image", ReturnImages)
        img_data = Image()

        img_data.width = s([]).width
        img_data.height = s([]).height
        img_data.encoding = s([]).encoding
        img_data.data = s([]).data

        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(img_data, "passthrough")
        cv_resized = cv2.resize(cv_image, (125, 125), interpolation = cv2.INTER_AREA)
        #cv2.imshow("Image window", cv_image)

        cv2.waitKey(1)

    return cv_resized


# Initialize simulation variables
T_MIN = 2
T_MAX = 4
INTERP = 100

# Initialize the time vector
t = time_vector(T_MAX, INTERP)

# weights file name
exp_string = FLAGS.experiment + '.' + FLAGS.init + '_init.' + str(FLAGS.num_conv_layers) + '_conv' + '.' + str(FLAGS.num_strides) + '_strides' + '.' + str(FLAGS.num_filters) + '_filters' + \
             '.' + str(FLAGS.num_fc_layers) + '_fc' + '.' + str(FLAGS.layer_size) + '_dim' + '.bt_dim_' + str(FLAGS.bt_dim) + '.mbs_' + str(FLAGS.meta_batch_size) + \
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


# Directory of the weights
vel_directory = FLAGS.log_dir + '/' + exp_string

# Initialize the graph, model and session
tf.set_random_seed(FLAGS.random_seed)
np.random.seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

graph = tf.Graph()

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

sess = tf.Session(graph=graph, config=tf_config)

network_config = {
    'num_filters': [FLAGS.num_filters] * FLAGS.num_conv_layers,
    'strides': [[1, 2, 2, 1]] * FLAGS.num_strides + [[1, 1, 1, 1]] * (FLAGS.num_conv_layers - FLAGS.num_strides),
    'filter_size': FLAGS.filter_size,
    'image_width': FLAGS.im_width,
    'image_height': FLAGS.im_height,
    'image_channels': FLAGS.num_channels,
    'n_layers': FLAGS.num_fc_layers,
    'layer_size': FLAGS.layer_size,
    'initialization': FLAGS.init,
}

print(FLAGS.demo_file)

# Initialize the data generator
data_generator = DataGenerator()
state_idx = data_generator.state_idx
img_idx = range(len(state_idx), len(state_idx) + FLAGS.im_height * FLAGS.im_width * FLAGS.num_channels)

# Initialize the model
model = MIL(data_generator._dU, state_idx=state_idx, img_idx=img_idx, network_config=network_config)
model.init_network(graph, prefix='Testing')

with graph.as_default():
    # Set up saver.
    saver = tf.train.Saver(max_to_keep=10)
    # Initialize variables.
    init_op = tf.global_variables_initializer()
    sess.run(init_op, feed_dict=None)
    # Start queue runners (used for loading videos on the fly)
    tf.train.start_queue_runners(sess=sess)

print(vel_directory)
model_file = tf.train.latest_checkpoint(vel_directory)
print(model_file)
ind1 = model_file.index('model')
resume_itr = int(model_file[ind1 + 6:])

# Load the model weights
print("Restoring model weights from " + model_file)
with graph.as_default():
    saver.restore(sess, model_file)

# Start the Simulation
robot = Robot()

# Initialize the IK solver
solver = robot.ik_init()

# Initialize the time vector
t = time_vector(T_MAX, INTERP)

# Move the torso of the body up
q = [0.27]
joint_names = ["torso"]
traj, d_traj, dd_traj = robot.simple_trajectory(joint_names, q, t)
goal = robot.generate_message(joint_names, traj, d_traj, dd_traj, t)
robot.arm_control(goal)
print("Robot Torso is up")

# Move the left hand of the robot out of the way
q = [0, 90, 0, 0, 0, 0, 0]
q = radiant(q)
joint_names = robot.joint_left_names
traj, d_traj, dd_traj = robot.simple_trajectory(joint_names, q, t)
goal = robot.generate_message(joint_names, traj, d_traj, dd_traj, t)
robot.arm_control(goal)

q = [0, 90, 80, 0, -130, -70, 0]
q = radiant(q)
joint_names = robot.joint_left_names
traj, d_traj, dd_traj = robot.simple_trajectory(joint_names, q, t)
goal = robot.generate_message(joint_names, traj, d_traj, dd_traj, t)
robot.arm_control(goal)
print("Left Hand is out of the way")

# Move the right hand of the robot in the home position
q = [-90, -70, 0, 0, -70, 0, 0]
q = radiant(q)
joint_names = robot.joint_right_names
traj, d_traj, dd_traj = robot.simple_trajectory(joint_names, q, t)
goal = robot.generate_message(joint_names, traj, d_traj, dd_traj, t)
robot.arm_control(goal)
print("Right Hand is in home position")

# Open the robot right gripper
grp_msg = robot.open_msg()
robot.gripper_control(grp_msg)
print("Gripper is open")

# Initialize the environment
env = Environment()

# Load the testing data, the scale and the bias
scale_file = FLAGS.scale_dir
scale, bias = load_scale_and_bias(scale_file)

files = glob.glob(os.path.join(FLAGS.demo_file, '*.pkl'))
all_ids = [int(f.split('/')[-1][:-4].split('_')[-1]) for f in files]
all_ids.sort()

# Traverse all the testing items
test_items = len(env.test_names)
trials = len(env.test_x)

# Variables to keep count of the success
num_success = 0
num_trials = 0
K = 2 # Control gain to amplify the output of the network

# DMP class initialization
my_dmp = DMP(20.0, 20, False)

# Get the phase from the time vector
s = my_dmp.phase(t)

# Get the Gaussian
psv = my_dmp.distributions(s)

# Initialize variables for the dmp evaluation
tau = t[-1]
dx_r = 0
x_r = radiant([-90, -70, 0, 0, -70, 0, 0])
x0 = x_r

# Variables to store the results
record_dataset = "/home/mtheofanidis/catkin_ws/src/Data/Unreal-Dataset/Final_Results_"
DATASET_ID = FLAGS.output + "_" + str(FLAGS.dataset_number)

# Initialize the Recorder
recorder = Recorder(DATASET_ID)
recorder.storing_directory = record_dataset

stored_trajectory = []
stored_position = []

#for index in range(0, test_items):
for index in [3]:

    # Print the items name
    print(env.test_names[index].get('name'))
    print('====================')

    # Empty the storing array
    recorder.empty_distance()

    # Load the demo for this item
    demoX, demoU, demo_gifs, demo_info = load_demo(str(all_ids[index]), FLAGS.test_dir, 0)

    # Concatenate demos in time
    demo_gif = np.array(demo_gifs)
    T, H, W, C = demo_gif.shape
    N = FLAGS.update_batch_size
    T = INTERP
    demo_gif = np.reshape(demo_gif, [N * T, H, W, C])
    demo_gif = np.array(demo_gif)[:, :, :, :3].transpose(0, 3, 2, 1).astype('float32') / 255.0
    demoVideo = demo_gif.reshape(1, N * T, -1)

    # Make a directory to store the demo images of the object
    recorder.make_image_directory(index)

    #for i in range(3, trials):
    for i in [8]:

        # Move the object in a random location on the table
        position, yaw, HT = env.move_test_object(i, index)
        print(position)

        # Calculate the ideal IK to get ideal final position and orientation
        [k, o] = env.generate_key_points(index, position, yaw)

        # Move the Robot near the object
        joint_names = robot.joint_right_names
        traj, d_traj, dd_traj, t, found, q_f, p_f, o_f = robot.generate_test_trajectory(joint_names, k, o, t)

        # Set Recorder parameters
        recorder.trajectory_id = i

        if index == 3 and i == 8:
            # Start the Recorder
            recorder.start()

            # Start Recording
            recorder.resume()
        else:
            # Start Recording
            recorder.resume()

        # Run the new trajectory from the network
        for stamp in range(0, len(t)-1):

            # Take a snapshot of the scene
            image = get_snapshot()

            # Resize the image
            image = np.expand_dims(image, 0).transpose(0, 3, 2, 1).astype('float32') / 255.0
            image = image.reshape((1, 1, -1))

            # Get the state of the robot
            q, dq, ddq = robot.get_joint_state(joint_names)

            # Resize the state vector
            q = q.reshape((1, 1, 7))

            # Pick an action according to the network
            action = sess.run(model.test_act_op,
                               {model.statea: demoX.dot(scale) + bias,
                                model.obsa: demoVideo,
                                model.actiona: demoU,
                                model.stateb: q.dot(scale) + bias,
                                model.obsb: image})

            if FLAGS.output is 'vel':

                # New joint position
                q_plus = q + K*action*(t[stamp+1]-t[stamp])
                q_new = q_plus[0][0]

            if FLAGS.output is 'dmp':

                # Gaussian weighted sum
                p_sum = 0
                p_div = 0

                # Time difference
                dt = t[stamp+1]-t[stamp]

                # Extract the weight vector
                w = action.reshape((my_dmp.ng, len(joint_names)))

                # Derive the forcing term
                sigma = (q_f - x0) * s[stamp]

                for j in range(my_dmp.ng):
                    p_sum += psv[j][stamp] * w[j]
                    p_div += psv[j][stamp]

                # Calculate the new control input
                f_target = (p_sum / p_div) * sigma

                # Calculate the new trajectory
                ddx_r = (my_dmp.a * (my_dmp.b * (q_f - x_r) - tau * dx_r) + f_target) / np.power(tau, 2)
                dx_r += (ddx_r * dt)
                x_r += (dx_r * dt)
                q_new = x_r

            # Send trajectory to the robot
            interval = time_vector(t[stamp+1]-t[stamp], 40)
            traj, d_traj, dd_traj = robot.simple_trajectory(joint_names, q_new, interval)
            goal = robot.generate_message(joint_names, traj, d_traj, dd_traj, interval)
            robot.arm_control(goal)

            # Check the distance of the robots end effector with respect to the table and the object
            [p_c, o_c] = env.find_final_transform("torso_lift_link", "r_gripper_tool_frame")

            # Store the robot state
            stored_position.append(np.array([q_new[0],q_new[1],q_new[2],q_new[3],q_new[4],q_new[5],q_new[6]]))
            stored_trajectory.append(np.array([p_c[0],p_c[1],p_c[2]]))

            # Calculate the distance from the table
            T = env.find_transform("odom_combined", "torso_lift_link")
            Z_T = -T[2, -1] - TABLE_HEIGHT
            distance_t = math.sqrt((p_c[2] - Z_T) ** 2)

            # Calculate the distance from the object
            distance_p = math.sqrt(((p_c[0]-p_f[0])**2)+((p_c[1]-p_f[1])**2)+((p_c[2]-p_f[2])**2))

            print(stamp)
            print('--------------------')
            print(distance_p)
            print('--------------------')

            # Evaluate wheras the robot is stuck
            comparison = np.array(q[0][0] - q_new)
            if all(i < 0.001 for i in comparison):
                print(comparison)
                print('The robot is stuck')
                break

            # Evaluate whereas the gripper is near the object
            if distance_p < FINAL_THRESHOLD:

                print('The gripper reached the object')
                break

            # Evaluate whereas the table was hit or not
            if distance_p < FINAL_THRESHOLD_T:
                print('The gripper hit the table')
                break

            # Evaluate whereas the results are an utter failure or not
            if distance_p > FAIL:
                print('Utter Failure')
                break

        # Store the distances
        recorder.store_distances(distance_p)

        # Record the state of the robot
        recorder.store_trajectories(record_dataset + DATASET_ID, stored_position, stored_trajectory,index)

        # Stop Recording
        recorder.pause()

        # Wait to finish recording
        rospy.sleep(rospy.Duration(1))

        # Create the gif
        recorder.make_gif(i)

        # Move the object back to its original location
        env.move_test_object_back(index)

        # Move the robot to home position
        print("Moving the robot arm to its home position")
        q = radiant([-90, -70, 0, 0, -70, 0, 0])
        joint_names = robot.joint_right_names
        traj, d_traj, dd_traj = robot.simple_trajectory(joint_names, q, t)
        goal = robot.generate_message(joint_names, traj, d_traj, dd_traj, t)
        robot.arm_control(goal)

    # Record the performance
    recorder.store_performance(record_dataset + DATASET_ID, index)

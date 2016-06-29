#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author:  <yao62995@gmail.com> 

import os
import re
import time
import json
import random
from collections import deque
import numpy as np
import tensorflow as tf

from util.AI_logger import logger
from AI_import import corpus, board_to_stream
from network.transform import distorted_inputs
from AI_renju import RenjuGame
from Renju import parser_argument, ModelRPC

train_dir = "./policy_rl_multi_gpu_models/"

planes = 3
filters = 192
layers = 5
board_size = 15

BATCH_SIZE = 128
TOWER_NAME = "tower"
NUM_CLASSES = 225
NUM_EPOCHS_PER_DECAY = 100
INITIAL_LEARNING_RATE = 0.00025  # 0.125 (epoch:1-25)=> 0.025 (epoch: 26-)
LEARNING_RATE_DECAY_FACTOR = 0.5
MOVING_AVERAGE_DECAY = 0.9999
GPU_MEMERY_ALLOCATE = 0.4

gpu_num = 4
session = None
saver = None


def _activation_summary(x):
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(name, shape,
                           tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def one_hot_encoding(labels, num_classes, scope=None):
    """Transform numeric labels into onehot_labels.

    Args:
      labels: [batch_size] target labels.
      num_classes: total number of classes.
      scope: Optional scope for op_scope.
    Returns:
      one hot encoding of the labels.
    """
    with tf.op_scope([labels], scope, 'OneHotEncoding'):
        batch_size = labels.get_shape()[0]
        indices = tf.expand_dims(tf.range(0, batch_size), 1)
        labels = tf.cast(tf.expand_dims(labels, 1), indices.dtype)
        concated = tf.concat(1, [indices, labels])
        onehot_labels = tf.sparse_to_dense(
            concated, tf.pack([batch_size, num_classes]), 1.0, 0.0)
        onehot_labels.set_shape([batch_size, num_classes])
        return onehot_labels


def conv2d(x, kernel_shape, variable_scope, stride=1, stddev=1e-2, padding="SAME"):
    with tf.variable_scope(variable_scope) as scope:
        kernel = _variable_with_weight_decay('weights', shape=kernel_shape,
                                             stddev=stddev, wd=1e-4)
        conv = tf.nn.conv2d(x, kernel, strides=[1, stride, stride, 1], padding=padding)
        biases = _variable_on_cpu('biases', [kernel_shape[-1]], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv)
    return conv


def full_connect(x, W_shape, variable_scope, stddev=0.04):
    with tf.variable_scope(variable_scope) as scope:
        weights = _variable_with_weight_decay('weights', shape=W_shape,
                                              stddev=stddev, wd=0.004)
        biases = _variable_on_cpu('biases', [W_shape[-1]], tf.constant_initializer(0.1))
        fc = tf.nn.relu(tf.matmul(x, weights) + biases, name=scope.name)
        _activation_summary(fc)
    return fc


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def inference(_input):
    # first conv1
    conv1 = conv2d(_input, (8, 8, planes, filters), "conv_1", stride=1)
    # norm1
    norm1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm_1')
    # conv2 ~ conv_k
    pre_layer = norm1
    for i in xrange(layers):
        conv_k = conv2d(pre_layer, (5, 5, filters, filters), "conv_%d" % (i + 2), stride=1)
        norm2 = tf.nn.lrn(conv_k, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm_%d' % (i + 2))
        pre_layer = norm2
    # last layer
    conv_n = conv2d(pre_layer, (3, 3, filters, 32), "conv_n", stride=1)
    norm_n = tf.nn.lrn(conv_n, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm_n')

    reshape = tf.reshape(norm_n, [-1, board_size * board_size * 32])
    # dim = reshape.get_shape()[1].value
    fc1 = full_connect(reshape, (board_size * board_size * 32, 1024), "fc_1")

    with tf.variable_scope("out") as scope:
        weights = _variable_with_weight_decay('weights', shape=(1024, board_size * board_size),
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [board_size * board_size], tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(fc1, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)
    return softmax_linear


def loss(logits, labels, batch_target):
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='cross_entropy_per_example')
    # predict_act = tf.reduce_sum(tf.mul(logits, one_hot_labels), reduction_indices=1)
    predict_loss_mean = tf.reduce_mean(tf.square(batch_target - cross_entropy), name='cross_entropy')

    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    #     logits, one_hot_encoding(labels, NUM_CLASSES), name='cross_entropy_per_example')
    #
    # cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', predict_loss_mean)
    # accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))
    tf.add_to_collection('accuracy', accuracy)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def tower_loss(scope, batch_x, batch_y, batch_target):
    """Calculate the total loss on a single tower running the CIFAR model.

    Args:
      scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'

    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """
    # file_records = "./data/renju_planes.tfrecords"
    # states, labels = distorted_inputs(file_records=file_records, shape=(board_size, board_size, planes),
    #                                   batch_size=BATCH_SIZE,
    #                                   num_epochs=200, num_threads=5, num_examples_per_epoch=1024)
    # states = tf.cast(states, tf.float32)
    # labels = tf.cast(labels, tf.float32)

    # Build inference Graph.
    logits = inference(batch_x)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = loss(logits, batch_y, batch_target)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', l.op.name)
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(loss_name + ' (raw)', l)
        tf.scalar_summary(loss_name, loss_averages.average(l))

    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss, logits


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def param_serierlize(param_file, param_dict):
    open(param_file, "w").write(json.dumps(param_dict))


def param_unserierlize(param_file, init_params=None):
    if os.path.exists(param_file):
        jd = json.loads(open(param_file, 'r').read())
    else:
        jd = init_params
    return jd


def restore_model(sess, model_dir, saver, model_file=None):
    if model_file is not None:
        model_file_path = "%s/%s" % (model_dir, model_file)
        saver.restore(sess, model_file_path)
        print("Successfully loaded:", model_file_path)
    else:
        checkpoint = tf.train.get_checkpoint_state(model_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")


def save_model(sess, model_dir, saver, prefix, global_step=None):
    checkpoint_filename = saver.save(sess, model_dir + "/" + prefix, global_step=global_step)
    return checkpoint_filename


def network(epochs=200, predict=False):
    param_file = "%s/param.json" % train_dir
    params = param_unserierlize(param_file, init_params={"global_step": 0})
    global_step_val = int(params["global_step"])
    """Train for a number of steps."""
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(global_step_val), trainable=False)

        # Calculate the learning rate schedule.
        num_batchs_per_epochs = corpus.num_batchs_per_epochs(BATCH_SIZE)
        print("num_batches_per_epoch: %d" % num_batchs_per_epochs)
        decay_steps = int(num_batchs_per_epochs / gpu_num * NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)

        # Create an optimizer that performs gradient descent.
        # opt = tf.train.GradientDescentOptimizer(lr)
        opt = tf.train.AdamOptimizer(lr)

        # Calculate the gradients for each model tower.
        tower_grads = []
        tower_acc = []
        tower_feeds = []
        tower_logits = []
        for i in xrange(gpu_num):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
                    # all towers.
                    batch_input = tf.placeholder(tf.float32, [None, board_size, board_size, planes])
                    batch_labels = tf.placeholder(tf.float32, shape=[None])
                    batch_target = tf.placeholder(tf.float32, shape=[None])
                    tower_feeds.append((batch_input, batch_labels, batch_target))
                    loss, logits = tower_loss(scope, batch_input, batch_labels, batch_target)
                    tower_logits.append(logits)

                    # all accuracy
                    tower_acc.append(tf.get_collection('accuracy', scope)[0])

                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    # Retain the summaries from the final tower.
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    # Calculate the gradients for the batch of data on this CIFAR tower.
                    grads = opt.compute_gradients(loss)

                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads)

        # average accuracy
        accuracy = tf.add_n(tower_acc) / len(tower_acc)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)


        # Add a summary to track the learning rate.
        summaries.append(tf.scalar_summary('learning_rate', lr))

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(
                    tf.histogram_summary(var.op.name + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.histogram_summary(var.op.name, var))

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Create a saver.
        # saver = tf.train.Saver(tf.all_variables())
        saver = tf.train.Saver()

        # Build the summary operation from the last tower summaries.
        summary_op = tf.merge_summary(summaries)

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_MEMERY_ALLOCATE)
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            gpu_options=gpu_options)
        )
        sess.run(init)

        # restore model
        restore_model(sess, train_dir, saver)
        if predict:
            return sess, saver

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        graph_def = sess.graph.as_graph_def(add_shapes=True)
        summary_writer = tf.train.SummaryWriter(train_dir,
                                                graph_def=graph_def)
        return sess, saver, summary_writer, train_op, loss, accuracy, global_step, lr, tower_feeds, tower_logits


def train_rl_network(batch_games=128, save_step=10000,
                     max_model_pools=5, init_epsilon=0.5, final_epsilon=0.01, explore=1000000,
                     action_repeat=32, mini_batch_size=64):
    """
        data set from self-play
    :return:
    """
    args = parser_argument().parse_args()
    rpc = ModelRPC(args)
    game = RenjuGame()
    batch_states, batch_actions, batch_rewards = deque(), deque(), deque()
    mini_batch_states, mini_batch_actions, mini_batch_rewards = [0] * mini_batch_size, [0] * mini_batch_size, [
        0] * mini_batch_size
    model_pools = []
    param_file = "%s/param.json" % train_dir
    params = param_unserierlize(param_file, init_params={"global_step": 0, "epsilon": init_epsilon})
    global_step_val, epsilon = params["global_step"], params["epsilon"]
    # load model
    sess, saver, summary_writer, train_op, loss, accuracy, global_step, lr, tower_feeds, tower_logits = network()
    train_step = 0
    while True:
        start_time = time.time()
        # choose policy network for opponent player from model pools
        if train_step % 10 == 0:
            if len(model_pools) > 0:
                model_file = random.choice(model_pools)
            else:
                model_file = None
            rpc.switch_model("policy_rl", model_file=model_file)
        while len(batch_states) < batch_games:
            # opponent_policy = self.load_history_policy_model(model_file)
            black_opponent = random.choice([True, False])
            # reset game
            game.reset_game()
            # simulate game by current parameter
            states, actions, rewards = [], [], []
            state = game.step_games(None)
            while True:  # loop current game
                # self-play, current model V.S. history model
                if (black_opponent and game.player == RenjuGame.PLAYER_BLACK) \
                        or (not black_opponent and game.player == RenjuGame.PLAYER_WHITE):
                    predict_probs = rpc.policy_rl_rpc(board_to_stream(game.board), game.get_player_name())
                else:  # current player
                    predict_probs = sess.run([tower_logits[0]], feed_dict={tower_feeds[0][0]: [state]})[0][0]
                if random.random() < epsilon:  # random choose action
                    action = game.weighted_choose_action(predict_probs)
                else:
                    action = game.choose_action(predict_probs)
                if action is None:
                    final_reward = 0
                    break
                # step game
                state_n, reward_n, terminal_n = game.step_games(action)
                # store (state, action)
                states.append(state)
                actions.append(action)
                # set new states
                state = state_n
                if terminal_n:
                    final_reward = reward_n
                    break
                # check whether game drawn
                if game.random_action() is None:  # game drawn, equal end, reward=0
                    final_reward = 0
                    logger.info("game drawn, so amazing...")
                    break
            # store (reward)
            for step in xrange(len(states)):
                if step % 2 == 0:
                    rewards.append(final_reward)
                else:
                    rewards.append(-final_reward)
            # store states of ith game
            batch_states.append(states)
            batch_actions.append(actions)
            batch_rewards.append(rewards)
        # fit model by mini batch
        avg_loss, avg_acc = 0.0, 0.0
        for _ in xrange(action_repeat / gpu_num):
            train_step += 1
            feeds = {}
            for gpu_id in xrange(gpu_num):
                for idx in xrange(mini_batch_size):
                    game_idx = random.randint(0, len(batch_states) - 1)
                    game_time_step_idx = random.randint(0, len(batch_states[game_idx]) - 1)
                    mini_batch_states[idx] = batch_states[game_idx][game_time_step_idx]
                    mini_batch_actions[idx] = batch_actions[game_idx][game_time_step_idx]
                    mini_batch_rewards[idx] = batch_rewards[game_idx][game_time_step_idx]
                feeds[tower_feeds[gpu_id][0]] = mini_batch_states
                feeds[tower_feeds[gpu_id][1]] = mini_batch_actions
                feeds[tower_feeds[gpu_id][2]] = mini_batch_rewards
            _, global_step_val, loss_val, acc_val = sess.run([train_op, global_step, loss, accuracy], feed_dict=feeds)
            avg_loss += loss_val
            avg_acc += acc_val
            # update epsilon
            if epsilon > final_epsilon:
                epsilon -= (init_epsilon - final_epsilon) / explore
        avg_loss /= action_repeat
        avg_acc /= action_repeat
        batch_states.popleft()
        batch_actions.popleft()
        batch_rewards.popleft()

        global_step_val = int(global_step_val)
        elapsed_time = int(time.time() - start_time)
        logger.info(
            "train policy rl network, step=%d, epsilon=%.5f, loss=%.6f, acc=%.6f, time=%d(sec)" %
            (train_step, epsilon, avg_loss, avg_acc, elapsed_time))
        # save model
        if train_step % save_step == 0:
            params["global_step"], params["epsilon"] = global_step_val, epsilon
            param_serierlize(param_file, params)
            model_file = save_model(sess, train_dir, saver,
                                    "policy_rl_step_%d" % train_step,
                                    global_step=global_step_val)
            logger.info("save policy rl model, file=%s" % model_file)
            model_file = model_file[len(train_dir):]
            # add history model to pool
            model_pools.append(model_file)
            if len(model_pools) > max_model_pools:  # pop head when model pools exceed
                model_pools.pop(0)
            logger.info("model pools has files: [%s]" % (", ".join(model_pools)))


if __name__ == "__main__":
    train_rl_network()

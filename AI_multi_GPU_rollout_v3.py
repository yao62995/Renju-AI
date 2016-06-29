#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author:  <yao62995@gmail.com> 

import os
import re
import time
import json
import numpy as np
import tensorflow as tf

from util.AI_logger import logger
from AI_import import corpus
from network.transform import distorted_inputs

train_dir = "./policy_rollout_multi_gpu_models/"

planes = 3
board_size = 15

BATCH_SIZE = 128
TOWER_NAME = "tower"
NUM_CLASSES = 225
NUM_EPOCHS_PER_DECAY = 20
INITIAL_LEARNING_RATE = 0.0025  # 0.125 (epoch:1-25)=> 0.025 (epoch: 26-)
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
                                             stddev=stddev, wd=0.0)
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
    fc1 = full_connect(_input, (15*15*planes, 15*15*planes), "fc_1")
    # fc_n = full_connect(fc1, (1024, 512), "fc_n")

    with tf.variable_scope("out") as scope:
        weights = _variable_with_weight_decay('weights', shape=(15*15*planes, board_size * board_size),
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [board_size * board_size], tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(fc1, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)
    return softmax_linear


def loss(logits, labels):
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='cross_entropy_per_example')
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    #     logits, one_hot_encoding(labels, NUM_CLASSES), name='cross_entropy_per_example')

    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    # accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))
    tf.add_to_collection('accuracy', accuracy)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def tower_loss(scope, batch_x, batch_y):
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
    _ = loss(logits, batch_y)

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
    return total_loss


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


def train(epochs=200, predict=False):
    param_file = "%s/param.json" % train_dir
    params = param_unserierlize(param_file, init_params={"epoch": 0, "global_step": 0})
    global_epoch, global_step_val = int(params["epoch"]), int(params["global_step"])
    """Train for a number of steps."""
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(global_step_val), trainable=False)

        # Calculate the learning rate schedule.
        num_batchs_per_epochs = int(corpus.num_batchs_per_epochs(BATCH_SIZE))
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
        for i in xrange(gpu_num):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
                    # all towers.
                    batch_input = tf.placeholder(tf.float32, [None, 15*15*planes])
                    batch_labels = tf.placeholder(tf.float32, shape=[None])
                    tower_feeds.append((batch_input, batch_labels))
                    loss = tower_loss(scope, batch_input, batch_labels)

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

        avg_loss, avg_acc = [], []
        epochs_step = global_epoch + 1
        step = 0
        while epochs_step <= (global_epoch + epochs):
            step += gpu_num
            start_time = time.time()
            # _, loss_value, acc_value, global_step_val = sess.run([train_op, loss, accuracy, global_step])
            feeds = {}
            for idx in xrange(gpu_num):
                samples = corpus.next_fetch_rows(BATCH_SIZE)
                feature = np.array([sample[0].get_states(flatten=True) for sample in samples], dtype=np.float32)
                labels = np.array([sample[1] for sample in samples], dtype=np.float32)
                feeds[tower_feeds[idx][0]] = feature
                feeds[tower_feeds[idx][1]] = labels
            _, loss_value, acc_value, global_step_val, learn_rating = sess.run(
                [train_op, loss, accuracy, global_step, lr],
                feed_dict=feeds)
            elapsed_time = int((time.time() - start_time) * 1000)

            avg_loss.append(loss_value)
            avg_acc.append(acc_value)

            global_step_val = int(global_step_val)
            if global_step_val % 10 == 0:
                logger.info(
                    "train policy rollout multi_GPU network, epoch=%d, step=%d, loss=%.6f, acc=%.6f, lr=%.6f, time=%d(ms)" % (
                        epochs_step, step, loss_value, acc_value, learn_rating, elapsed_time))

            # if global_step_val % 100 == 0:
            #     summary_str = sess.run(summary_op)
            #     summary_writer.add_summary(summary_str, step)
            if step > num_batchs_per_epochs:
                step = step % num_batchs_per_epochs
                epochs_step += 1
                average_loss = sum(avg_loss) / len(avg_loss)
                average_acc = sum(avg_acc) / len(avg_acc)
                avg_loss, avg_acc = [], []

                logger.info("train policy rollout multi_GPU network, epochs=%d, average_loss=%.7f, average_acc=%.7f" %
                            (epochs_step, average_loss, average_acc))
                # Save the model checkpoint periodically.
                if epochs_step % 5 == 0:
                    param_serierlize(param_file, {"epoch": int(epochs_step), "global_step": int(global_step_val)})
                    filename = save_model(sess, train_dir, saver,
                                          "policy_rollout_epoch_%d" % epochs_step,
                                          global_step=global_step_val)
                    logger.info("save policy rollout multi_GPU model: %s" % filename)


if __name__ == "__main__":
    train()
    # predict_model()

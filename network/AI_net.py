#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author:  <yao62995@gmail.com>

import os
import re
import simplejson as json
import tensorflow as tf

TOWER_NAME = "tower"


def weight_variable(shape, stddev=1.0, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)


def weight_variable_v2(shape, stddev=1.0, name=None):
    initial = tf.random_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)


def bias_variable_v2(shape, stddev=1.0, name=None):
    initial = tf.random_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def get_variable_name(prefix=None, var_num=0):
    if prefix is not None:
        return "%s_%d" % (prefix, var_num), var_num + 1
    else:
        return None, var_num


def create_policy_network(_input, planes, filters=192, board_size=15, layers=5):
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


def create_value_network(planes, ps_device, worker_device, filters=192, board_size=15, layers=5, name_prefix=None):
    variable_num = 0
    with tf.device(ps_device):
        var_name, variable_num = get_variable_name(name_prefix, variable_num)
        W_conv1 = weight_variable((5, 5, planes, filters), stddev=0.1, name=var_name)
        var_name, variable_num = get_variable_name(name_prefix, variable_num)
        b_conv1 = bias_variable([filters], name=var_name)

        W_conv_k, b_conv_k = [], []
        for _ in range(layers):
            var_name, variable_num = get_variable_name(name_prefix, variable_num)
            W_conv_k.append(weight_variable((3, 3, filters, filters), stddev=0.1, name=var_name))
            var_name, variable_num = get_variable_name(name_prefix, variable_num)
            b_conv_k.append(bias_variable([filters]))
        var_name, variable_num = get_variable_name(name_prefix, variable_num)
        W_conv_n = weight_variable((1, 1, filters, 1), stddev=0.1, name=var_name)
        var_name, variable_num = get_variable_name(name_prefix, variable_num)
        b_conv_n = bias_variable([1])

        var_name, variable_num = get_variable_name(name_prefix, variable_num)
        W_fc1 = weight_variable([board_size * board_size, 256], stddev=0.1, name=var_name)
        var_name, variable_num = get_variable_name(name_prefix, variable_num)
        b_fc1 = bias_variable([256], name=var_name)

        var_name, variable_num = get_variable_name(name_prefix, variable_num)
        W_fc2 = weight_variable([256, 3], stddev=0.1, name=var_name)
        var_name, variable_num = get_variable_name(name_prefix, variable_num)
        b_fc2 = bias_variable([3], name=var_name)

        # input
        _input = tf.placeholder("float", [None, board_size, board_size, planes])

    with tf.device(worker_device):
        # first cnn layer
        h_conv1 = tf.nn.relu(conv2d(_input, W_conv1, stride=1) + b_conv1)
        # middle cnn layers
        pre_input = h_conv1
        for i in range(layers):
            h_conv_k = tf.nn.relu(conv2d(pre_input, W_conv_k[i], stride=1) + b_conv_k[i])
            pre_input = h_conv_k
        # last cnn layers
        h_conv_n = tf.nn.relu(conv2d(h_conv_k, W_conv_n, stride=1) + b_conv_n)

        # softmax
        h_flat1 = tf.reshape(h_conv_n, [-1, board_size * board_size])
        h_fc1 = tf.nn.relu(tf.matmul(h_flat1, W_fc1) + b_fc1)

        # _output = tf.tanh(tf.matmul(h_fc1, W_fc2) + b_fc2)
        _output = tf.matmul(h_fc1, W_fc2) + b_fc2

    return _input, _output


def create_rollout_network(_input, planes, board_size=15):
    fc1 = full_connect(_input, (15*15*planes, 15*15*planes), "fc_1")

    with tf.variable_scope("out") as scope:
        weights = _variable_with_weight_decay('weights', shape=(15*15*planes, board_size * board_size),
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [board_size * board_size], tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(fc1, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)
    return softmax_linear


class SuperNetwork(object):
    def __init__(self, model_dir=""):
        # set session
        # self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))
        # load model if exist
        self.model_dir = model_dir
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        self.saver = tf.train.Saver(max_to_keep=None)
        # self.restore_model(model_file=model_file)
        self.param_file = "%s/params.json" % self.model_dir
        self.session = None

    def set_session(self, sess):
        self.session = sess

    def close(self):
        # frees all resources associated with the session
        self.session.close()

    def param_serierlize(self, param_dict):
        open(self.param_file, "w").write(json.dumps(param_dict))

    def param_unserierlize(self, init_params=None):
        if os.path.exists(self.param_file):
            jd = json.loads(open(self.param_file, 'r').read())
        else:
            jd = init_params
        return jd

    def restore_model(self, model_file=None):
        if model_file is not None:
            model_file_path = "%s/%s" % (self.model_dir, model_file)
            self.saver.restore(self.session, model_file_path)
            print("Successfully loaded:", model_file_path)
            return True
        else:
            checkpoint = tf.train.get_checkpoint_state(self.model_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.session, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
                return True
            else:
                print("Could not find old network weights")
        return False

    def save_model(self, prefix, global_step=None):
        checkpoint_filename = self.saver.save(self.session, self.model_dir + "/" + prefix, global_step=global_step)
        return checkpoint_filename


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


def loss(logits, labels):
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    # accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))
    tf.add_to_collection('accuracy', accuracy)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def tower_loss(scope, inference, states, labels):
    # Build inference Graph.
    logits = inference(states)
    tf.add_to_collection('logits', logits)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = loss(logits, labels)

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

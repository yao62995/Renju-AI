#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author:  <yao62995@gmail.com> 

import time
import random

import tensorflow as tf

from util.AI_logger import logger
from network import AI_net
from AI_renju import one_hot_action


class PolicyDLNetwork(AI_net.SuperNetwork):
    def __init__(self, planes, corpus, policy_args, filters=192, board_size=15, model_dir="./policy_dl_models",
                 device="gpu", gpu=1, optimizer="sgd", learn_rate=1e-6, distributed_train=False):
        self.board_size = board_size
        # init network
        if distributed_train:
            ps_device = "/job:ps/task:0/cpu:0"
            worker_device = "/job:worker/task:%d/gpu:%d" % (policy_args.task_index, policy_args.gpu_id)
        else:
            # ps_device = "/cpu:0"
            if device == "cpu":
                worker_device = "/cpu:0"
            else:
                worker_device = "/gpu:%d" % gpu
            ps_device = worker_device
        self.tf_var = dict({"in": tf.placeholder(tf.float32, [None, board_size, board_size, planes])})
        with tf.device(worker_device):
            with tf.name_scope('tower_%d' % 0) as scope:
                self.tf_var["out"] = AI_net.create_policy_network(self.tf_var["in"],
                                                                  planes, filters=filters, board_size=self.board_size,
                                                                  layers=5)
        # super init
        AI_net.SuperNetwork.__init__(self, model_dir=model_dir)
        history_step = int(self.param_unserierlize(init_params={"global_step": 0})["global_step"])
        with tf.device(ps_device):
            self.global_step = tf.Variable(history_step)
        # loss function
        with tf.device(worker_device):
            # num_batchs = corpus.num_batchs_per_epochs(policy_args.policy_dl_batch_size)
            # decay_steps = policy_args.policy_dl_epochs_per_decay * num_batchs
            # learn_rate = tf.train.exponential_decay(learn_rate, self.global_step, decay_steps,
            #                                         policy_args.policy_dl_decay_rate, staircase=True)
            self.tf_var["lr"] = tf.Variable(learn_rate)
            self.loss_function(optimizer, learn_rate)
        # register all variable
        # self.session.run(tf.initialize_all_variables())

    def loss_function(self, optimizer, learn_rate):
        # loss model
        self.tf_var["target"] = tf.placeholder("float", [None, self.board_size * self.board_size])
        self.tf_var["cost"] = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(self.tf_var["out"], self.tf_var["target"])
        )
        # optimizer
        if optimizer == "sgd":
            self.tf_var["optimizer"] = \
                tf.train.GradientDescentOptimizer(learn_rate).minimize(self.tf_var["cost"], global_step=self.global_step)
        elif optimizer == "adam":
            self.tf_var["optimizer"] = \
                tf.train.AdamOptimizer(learn_rate).minimize(self.tf_var["cost"], global_step=self.global_step)
        elif optimizer == "rmsProb":
            self.tf_var["optimizer"] = \
                tf.train.RMSPropOptimizer(learn_rate).minimize(self.tf_var["cost"], global_step=self.global_step)
        else:
            logger.error("not found optimizer=%s" % optimizer, to_exit=True)
        # evaluate
        correct_pred = tf.equal(tf.argmax(self.tf_var["out"], 1), tf.argmax(self.tf_var["target"], 1))
        self.tf_var["accuracy"] = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def fit(self, state, action, fetch_info=False):
        fetch_var = [self.tf_var["optimizer"], self.global_step]
        if fetch_info:
            fetch_var.extend([self.tf_var["cost"], self.tf_var["accuracy"], self.tf_var["lr"]])
        fetch_status = self.session.run(fetch_var,
                                        feed_dict={self.tf_var["in"]: state, self.tf_var["target"]: action})
        return fetch_status

    def predict(self, batch_states):
        predict_out = self.tf_var["out"].eval(session=self.session, feed_dict={self.tf_var["in"]: batch_states})
        return predict_out

    def train_policy_network(self, corpus, epochs=20, batch_size=64, save_step=5):
        """
        :param states: [array(15, 15, planes)]
        :param actions:  [one_hot_list(0~225),]
        :return:
        """
        start_time = time.time()
        params = self.param_unserierlize(init_params={"epoch": 0, "global_step": 0})
        global_epoch, global_step = int(params["epoch"]), int(params["global_step"])
        epochs_step = global_epoch
        while epochs_step < (global_epoch + epochs):
            epochs_step += 1
            average_loss = 0.0
            average_acc = 0.0
            local_step = 0
            corpus.shuffle_datas()
            elapsed_time = 0.0
            for samples in corpus.iterator_fetch_rows(batch_size):
                sample_states = [sample[0].get_states() for sample in samples]
                sample_actions = [one_hot_action(sample[1]) for sample in samples]
                start_time = time.time()
                fetch_status = self.fit(sample_states, sample_actions, fetch_info=True)
                elapsed_time += int((time.time() - start_time) * 1000)
                _, global_step, loss, acc, lr = fetch_status
                # record loss
                local_step += 1
                average_loss += loss
                average_acc += acc
                # record time
                if global_step % 8 == 0:
                    logger.info("train policy dl network, epochs=%d, global_step=%d, loss=%.7f, avg_loss=%.7f, acc=%.7f, avg_acc=%.7f, lr=%.7f, time=%d(ms)" %
                            (epochs_step, global_step, loss, average_loss / local_step, acc, average_acc / local_step, lr, elapsed_time))
                    elapsed_time = 0
            logger.info("train policy dl network, epochs=%d, average_loss=%.7f, average_acc=%.7f" %
                        (epochs_step, average_loss / local_step, average_acc / local_step))
            if epochs_step % save_step == 0:  # save model
                self.param_serierlize({"epoch": int(epochs_step), "global_step": int(global_step)})
                filename = self.save_model("policy_dl_epoch_%d" % epochs_step, global_step=global_step)
                logger.info("save policy dl model: %s" % filename)





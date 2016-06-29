#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author:  <yao62995@gmail.com> 

import os
import math
import time
import random
import cPickle
import numpy as np
import tensorflow as tf

from network import AI_net
from AI_renju import RenjuGame
from util.AI_logger import logger
from AI_import import board_to_stream


class ValueNetwork(AI_net.SuperNetwork):
    def __init__(self, planes, args, phase=1, filters=192, board_size=15, model_dir="./value_net_models",
                 model_file=None,
                 device="gpu", gpu=1, optimizer="sgd", learn_rate=1e-6, distributed_train=False):
        self.board_size = board_size
        self.phase = phase
        self.planes = planes
        # init network
        if distributed_train:
            ps_device = "/job:ps/task:0/cpu:0"
            worker_device = "/job:worker/task:%d/gpu:%d" % (args.task_index, args.gpu_id)
        else:
            ps_device = "/cpu:0"
            if device == "cpu":
                worker_device = "/cpu:0"
            else:
                worker_device = "/gpu:%d" % gpu
        self.tf_var = dict()
        self.tf_var["in"], self.tf_var["out"] = AI_net.create_value_network(
            planes, ps_device, worker_device, filters=filters, board_size=self.board_size, name_prefix="value_net")
        # super init
        AI_net.SuperNetwork.__init__(self, model_dir=model_dir)
        history_step = int(self.param_unserierlize(init_params={"global_step": 0})["global_step"])
        with tf.device(ps_device):
            self.global_step = tf.Variable(history_step)
        # loss function
        with tf.device(worker_device):
            self.loss_function(optimizer, learn_rate, args.values_net_batch_size)

    def loss_function(self, optimizer, learn_rate, batch_size):
        self.tf_var["target"] = tf.placeholder("float", [None])
        self.tf_var["cost"] = tf.reduce_sum(tf.pow(self.tf_var["out"] - self.tf_var["target"], 2) / (2 * batch_size))
        if optimizer == "sgd":
            self.tf_var["optimizer"] = tf.train.GradientDescentOptimizer(learn_rate).minimize(self.tf_var["cost"], global_step=self.global_step)
        elif optimizer == "adam":
            self.tf_var["optimizer"] = tf.train.AdamOptimizer(learn_rate).minimize(self.tf_var["cost"], global_step=self.global_step)
        elif optimizer == "rmsProb":
            self.tf_var["optimizer"] = tf.train.RMSPropOptimizer(learn_rate).minimize(self.tf_var["cost"], global_step=self.global_step)
        else:
            logger.error("not found optimizer=%s" % optimizer, to_exit=True)

    def fit(self, state, target, fetch_info=False):
        fetch_var = [self.tf_var["optimizer"], self.global_step]
        if fetch_info:
            fetch_var.extend([self.tf_var["cost"]])
        fetch_status = self.session.run(fetch_var,
                                        feed_dict={self.tf_var["in"]: state, self.tf_var["target"]: target})
        return fetch_status

    def predict(self, batch_states):
        predict_out = self.tf_var["out"].eval(session=self.session, feed_dict={self.tf_var["in"]: batch_states})
        return predict_out

    def train_value_network(self, rpc, sample_num=1000, max_time_steps=225,
                            epochs=20, batch_size=32):
        """
        :param policy_dl: policy network of deep learning
        :param policy_rl: policy network of reinforcement learning
        :return:
        """
        model_params = self.param_unserierlize(init_params={"global_step": 0, "global_epoch": 0})
        if sample_num > 0:  # create sample
            start_time = time.time()
            sample_file = "data/value_net_phase_%d_samples_%d.pkl" % (self.phase, sample_num)
            sample_games = sampling_for_value_network(rpc, sample_num, sample_file, max_time_steps=max_time_steps)
            elapsed_time = int((time.time() - start_time) * 1000)
            logger.info("sampling for value network, samples=%d, time=%d(ms)" % (sample_num, elapsed_time))
            cPickle.dump(sample_games, open(sample_file, "wb"), protocol=2)
            logger.info("save sample file: %s" % sample_file)
            model_params["sample_file"] = sample_file
            self.param_serierlize(model_params)
        else:  # load old sample
            if 'sample_file' not in model_params:
                logger.error("not found sample file", to_exit=True)
            sample_games = cPickle.load(open(model_params["sample_file"], 'rb'))
        epoch_step, train_step = model_params["global_epoch"], model_params["global_step"]
        while epoch_step < (model_params["global_epoch"] + epochs):
            start_time = time.time()
            epoch_step += 1
            random.shuffle(sample_games)
            avg_loss = 0.0
            for idx in xrange(0, len(sample_games), batch_size):
                end_idx = min(len(sample_games), idx + batch_size)
                mini_samples = sample_games[idx: end_idx]
                # transform sample data
                mini_states = [sampled_game.get_states(player_plane=True) for sampled_game, _ in mini_samples]
                mini_rewards = [sampled_reward for _, sampled_reward in mini_samples]
                fetch_status = self.fit(mini_states, mini_rewards, fetch_info=True)
                _, train_step, loss = fetch_status
                avg_loss += loss
                train_step = int(train_step)
                if train_step % 20 == 0:
                    elapsed_time = int((time.time() - start_time) * 1000)
                    logger.info(
                        "train value network, phase=%d, epoch=%d, step=%d, loss=%.7f, time=%d(ms)" %
                        (self.phase, epoch_step, train_step, loss, elapsed_time))
                    start_time = time.time()
            avg_loss /= math.ceil(len(sample_games) / batch_size)
            logger.info("train value network, phase=%d, epoch=%d, avg_loss=%.6f" % (self.phase, epoch_step, avg_loss))
            if epoch_step % 5 == 0:  # save model
                model_params["global_step"] = train_step
                model_params["global_epoch"] = epoch_step
                self.param_serierlize(model_params)
                model_file = self.save_model("value_net_phase_%d" % self.phase, global_step=model_params["global_step"])
                logger.info("save value network model, file=%s" % model_file)


def sampling_for_value_network(rpc, sample_num, sample_file, max_time_steps=225):
    """
    :param max_steps: max time steps in games
    :return:
    """
    sample_games = []
    if os.path.exists(sample_file):
        sample_games = cPickle.load(open(sample_file, 'rb'))
        logger.info("load sample file: %s, samples=%d" % (sample_file, len(sample_games)))
    sample_sets = set()  # used to check unique sample
    game = RenjuGame()
    record_policy_dl_boards = []
    # move step by policy dl
    game.reset_game()
    record_policy_dl_boards.append(game.replicate_game())
    while True:
        action = game.choose_action(
            rpc.policy_dl_rpc(board_to_stream(game.board), game.get_player_name()))
        if action is None:
            break
        state, _, terminal = game.step_games(action)
        if terminal:
            break
        record_policy_dl_boards.append(game.replicate_game())
    max_time_steps = min(max_time_steps, len(record_policy_dl_boards)) - 1
    # sample game
    while len(sample_games) < sample_num:
        sampled_game = None
        while True:  # loop to find legal sample
            flag_time_step = random.randint(1, max_time_steps)
            recorded_game = record_policy_dl_boards[flag_time_step - 1].replicate_game()
            random_action = recorded_game.random_action()
            if random_action is None:
                break
            random_state, _, terminal = recorded_game.step_games(random_action)
            if not terminal and not str(random_state) in sample_sets:
                sample_sets.add(str(random_state))
                break
        if random_action is None:  # invalid loop
            continue
        # move step by policy rl
        time_step = flag_time_step
        while True:  # simulate game by policy rl
            actions = rpc.policy_rl_rpc(board_to_stream(recorded_game.board), recorded_game.get_player_name())
            action = recorded_game.choose_action(actions)
            if action is None:  # game drawn
                sampled_reward = 0
                break
            state, reward, terminal = recorded_game.step_games(action)
            time_step += 1
            if time_step == (flag_time_step + 1):  # record board
                sampled_game = recorded_game.replicate_game()
            if terminal:  # record value
                sampled_reward = reward
                break
        if sampled_game is not None:
            sample_games.append((sampled_game, sampled_reward))
            logger.info("sample simulate, sample_step=%d, time_step=%d" % (len(sample_games), time_step))
        if len(sample_games) % 100 == 0:
            cPickle.dump(sample_games, open(sample_file, "wb"), protocol=2)
            logger.info("create value network sample, step=%d" % len(sample_games))
    return sample_games

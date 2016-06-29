#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author:  <yao62995@gmail.com> 

import time
import random
from collections import deque
import numpy as np
import tensorflow as tf

from util.AI_logger import logger
from network import AI_net
from AI_renju import RenjuGame, one_hot_action, transform_action
from AI_import import board_to_stream


class PolicyRLNetwork(AI_net.SuperNetwork):
    def __init__(self, planes, policy_args, phase=1, filters=192, board_size=15, model_dir="./policy_rl_models",
                 device="gpu", gpu=1, optimizer="sgd", learn_rate=1e-6, distributed_train=False):
        self.board_size = board_size
        self.phase = phase
        self.planes = planes
        self.filters = filters
        # init network
        if distributed_train:
            ps_device = "/job:ps/task:0/cpu:0"
            worker_device = "/job:worker/task:%d/gpu:%d" % (policy_args.task_index, policy_args.gpu_id)
        else:
            ps_device = "/cpu:0"
            if device == "cpu":
                worker_device = "/cpu:0"
            else:
                worker_device = "/gpu:%d" % gpu
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
            self.loss_function(optimizer, learn_rate)
        # register all variable
        # self.session.run(tf.initialize_all_variables())

    def loss_function(self, optimizer, learn_rate):
        self.tf_var["act"] = tf.placeholder("float", [None, self.board_size * self.board_size])
        self.tf_var["target"] = tf.placeholder("float", [None])
        predict_act = tf.reduce_sum(tf.mul(self.tf_var["out"], self.tf_var["act"]), reduction_indices=1)
        self.tf_var["cost"] = tf.reduce_mean(tf.square(self.tf_var["target"] - predict_act))
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
        correct_pred = tf.equal(tf.argmax(self.tf_var["out"], 1), tf.argmax(self.tf_var["act"], 1))
        self.tf_var["accuracy"] = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def fit(self, state, action, target, fetch_info=False):
        fetch_var = [self.tf_var["optimizer"], self.global_step]
        if fetch_info:
            fetch_var.extend([self.tf_var["cost"], self.tf_var["accuracy"]])
        fetch_status = self.session.run(fetch_var, feed_dict={
            self.tf_var["in"]: state,
            self.tf_var["act"]: action,
            self.tf_var["target"]: target,
        })
        return fetch_status

    def predict(self, batch_states):
        predict_out = self.tf_var["out"].eval(session=self.session, feed_dict={self.tf_var["in"]: batch_states})
        return predict_out

    def load_history_policy_model(self, model_file):
        AI_net.create_policy_network(3, "/cpu:0", "/cpu:0")
        history_policy = PolicyRLNetwork(self.planes, None, filters=self.filters, board_size=self.board_size,
                                         model_dir=self.model_dir, device="cpu", distributed_train=False)
        history_policy.session = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True))
        history_policy.session.run(tf.initialize_all_variables())
        history_policy.restore_model(model_file=model_file)
        return history_policy

    def train_policy_network(self, rpc, batch_games=128, save_step=50000,
                             max_model_pools=5, init_epsilon=0.5, final_epsilon=0.05, explore=1000000,
                             action_repeat=20, mini_batch_size=128):
        """
            data set from self-play
        :return:
        """
        game = RenjuGame()
        batch_states, batch_actions, batch_rewards = deque(), deque(), deque()
        mini_batch_states, mini_batch_actions, mini_batch_rewards = [0] * mini_batch_size, [0] * mini_batch_size, [0] * mini_batch_size
        model_pools = []
        params = self.param_unserierlize(init_params={"global_step": 0, "epsilon": init_epsilon})
        global_step_val, epsilon = params["global_step"], params["epsilon"]
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
                    if random.random() < epsilon:  # random choose action
                        action = game.random_action()
                    else:
                        if (black_opponent and game.player == RenjuGame.PLAYER_BLACK) \
                                or (not black_opponent and game.player == RenjuGame.PLAYER_WHITE):
                            action = game.choose_action(
                                rpc.policy_rl_rpc(board_to_stream(game.board), game.get_player_name()))
                        else:  # current player
                            action = game.choose_action(self.predict([state])[0])
                    # step game
                    state_n, reward_n, terminal_n = game.step_games(action)
                    # print "game=", batch_step, ", move=", transform_action(action)
                    # store (state, action)
                    states.append(state)
                    one_hot_act = one_hot_action(action)
                    actions.append(one_hot_act)
                    # set new states
                    state = state_n
                    if terminal_n:
                        final_reward = reward_n
                        # logger.info("winner=%s" % ("black" if reward_n > 0 else "white"))
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
            for _ in xrange(action_repeat):
                train_step += 1
                for idx in xrange(mini_batch_size):
                    game_idx = random.randint(0, len(batch_states) - 1)
                    game_time_step_idx = random.randint(0, len(batch_states[game_idx]) - 1)
                    mini_batch_states[idx] = batch_states[game_idx][game_time_step_idx]
                    mini_batch_actions[idx] = batch_actions[game_idx][game_time_step_idx]
                    mini_batch_rewards[idx] = batch_rewards[game_idx][game_time_step_idx]
                _, global_step_val, loss, acc = self.fit(mini_batch_states, mini_batch_actions, mini_batch_rewards, fetch_info=True)
                avg_loss += loss
                avg_acc += acc
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
                (global_step_val, epsilon, avg_loss, avg_acc, elapsed_time))
            # save model
            if train_step % save_step == 0:
                params["global_step"], params["epsilon"] = global_step_val, epsilon
                self.param_serierlize(params)
                model_file = self.save_model("policy_rl", global_step=global_step_val)
                logger.info("save policy dl model, file=%s" % model_file)
                model_file = model_file[len(self.model_dir):]
                # add history model to pool
                model_pools.append(model_file)
                if len(model_pools) > max_model_pools:  # pop head when model pools exceed
                    model_pools.pop(0)
                logger.info("model pools has files: [%s]" % (", ".join(model_pools)))

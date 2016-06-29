#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author:  <yao62995@gmail.com> 

import time
import math
import random
import numpy as np
import threading

from AI_renju import RenjuGame, transform_action
from util.AI_logger import logger
from AI_import import board_to_stream, stream_to_board
from util.AI_tools import Utility


def normalize_prior_probs(prior_probs):
    min_probs = min(prior_probs)
    # prior_probs = map(lambda prob: np.exp(prob - min_probs), prior_probs)
    prior_probs = map(lambda prob: prob - min_probs, prior_probs)
    sum_probs = sum(prior_probs)
    return map(lambda prob: prob / sum_probs, prior_probs)


class Node(object):
    def __init__(self, position, parent=None):
        self.position = position  # an instance of RenjuGame
        self.parent = parent
        self.edges = []  # edges of node
        self.child = []

    def child_num(self):
        return len(self.child)

    def generate_edges(self, prior_probs):
        """
        :param prior_prob: list of probability for all actions
        :return:
        """
        actions = self.position.legal_actions()
        node = self
        for action in actions:
            prior_prob = prior_probs[action]
            edge = Edge(node, action, prior_prob)
            self.edges.append(edge)
            self.child.append(None)

    def is_leaf(self):
        return self.child_num() == 0


class Edge(object):
    def __init__(self, node, action, prior_prob):
        # basic variants
        self.node = node
        self.action = action
        self.is_over = self.node.position.game_over(transform_action(self.action))

        # series of statistics
        # prior probability
        self.prior_prob = prior_prob
        # leaf evaluate and rollout rewards
        self.leaf_evaluate = 0.0
        self.rollout_rewards = 0.0
        # Monte-Carlo estimates of total action-value
        self.total_evaluate = 0.0
        self.total_rewards = 0.0
        self.action_value = 0.0

    def update_action_value(self, mix_lambda):
        evaluate_part = self.total_evaluate / self.leaf_evaluate
        reward_part = self.total_rewards / self.rollout_rewards
        sum_part = (evaluate_part + reward_part)
        if -1e-6 < sum_part < 1e-6:
            return
        self.action_value = (1 - mix_lambda) * (evaluate_part / sum_part) + mix_lambda * (reward_part / sum_part)

    def edge_bonus(self, explore_rate):
        sum_rollout_rewards = 0.0
        for _edge in self.node.edges:
            sum_rollout_rewards += _edge.rollout_rewards
        if sum_rollout_rewards == 0:
            return explore_rate * self.prior_prob
        sign = 1 if sum_rollout_rewards > 0 else -1
        bonus = sign * math.sqrt(sum_rollout_rewards * sign) / (1.0 + self.rollout_rewards)
        bonus *= explore_rate * self.prior_prob
        return bonus

    def edge_weight(self, explore_rate):
        return self.action_value + self.edge_bonus(explore_rate)


class MCTS(object):
    def __init__(self, rpc, visit_threshold=40, virtual_loss=0, explore_rate=1.0, mix_lambda=0.5):
        """
        :param rpc: an instance of AI_tools.ModelRPC
        :param visit_threshold(constant): visit count threshold
        :param explore_rate(constant): explore rate
        :param mix_lambda(constant): mixing lambda
        """
        # constant
        self.visit_threshold = visit_threshold
        self.virtual_loss = virtual_loss
        self.explore_rate = explore_rate
        self.mix_lambda = mix_lambda
        # network
        self.rpc = rpc

    def analysis(self, node, depth=0):
        if node is None:
            return
        for idx, edge in enumerate(node.edges):
            if edge.leaf_evaluate == 0.0 and edge.rollout_rewards == 0.0:
                continue
            print "--" * depth, "edge_%d_%s(a=%d, p=%.3f, av=%.3f, le=%.3f, rr=%.3f, te=%.3f, tr=%.3f, weight=%.4f)" % \
                                (
                                    depth, node.position.get_player_name(), edge.action, edge.prior_prob,
                                    edge.action_value,
                                    edge.leaf_evaluate,
                                    edge.rollout_rewards, edge.total_evaluate, edge.total_rewards,
                                    edge.edge_weight(self.explore_rate))
            self.analysis(node.child[idx], depth=depth + 1)

    def decision(self, node):
        # find game-over edge
        for idx, edge in enumerate(node.edges):
            if edge.is_over:
                return None, edge.action
        act_q_values = np.empty(node.child_num(), dtype=float)
        # [edge.edge_weight(self.explore_rate) for idx, edge in enumerate(node.edges)]
        for idx, edge in enumerate(node.edges):
            act_q_values[idx] = edge.edge_weight(self.explore_rate)
        # if node.position.player == RenjuGame.PLAYER_WHITE:  # min for white player
        #     act_q_values = -act_q_values
        best_edge_idx = np.argmax(act_q_values)
        if node.child[best_edge_idx] is None:
            child_node_position = node.position.replicate_game()
            child_node_position.step_games(node.edges[best_edge_idx].action)
            node.child[best_edge_idx] = Node(child_node_position, parent=node)
        child_node = node.child[best_edge_idx]
        action = node.edges[best_edge_idx].action
        print "choose edge idx:", best_edge_idx, "action: ", transform_action(action), ", value=", act_q_values[
            best_edge_idx]
        return child_node, action

    def simulation(self, root, time_limit=None, freq_limit=20):
        """
            simulation of MCTS
        :param root: an instance of Node
        :param simulation: number of simulation
        :return:
        """
        if time_limit is not None:
            start_time = time.time()
            while True:
                self.pipline(root)
                self.analysis(root)
                # check time escaped
                if int(time.time() - start_time) > time_limit:
                    break
        else:
            for _ in xrange(freq_limit):
                self.pipline(root)
                # self.analysis(root)
        return root

    def pipline(self, root, host=None):
        # selection
        leaf_node, select_track = Utility.timeit(lambda: self.selection(root), desc="selection")
        # check end
        if leaf_node.parent.edges[select_track[-1]].is_over:
            out_come = 1 if leaf_node.parent.position.player == RenjuGame.PLAYER_BLACK else -1
        else:
            # evaluation
            out_come = Utility.timeit(lambda: self.evaluation(leaf_node, select_track, host=host), desc="evaluation")
        print "out_come:", out_come
        # backup
        Utility.timeit(lambda: self.backup(root, select_track, out_come), desc="backup")
        # expansion
        Utility.timeit(lambda: self.expansion(leaf_node, select_track), desc="expansion")

    def selection(self, root):
        """
            selection phase
        :param root:
        :return:
        """
        select_track = []
        # tree traversal
        node, node_parent = root, None
        print "select track: [",
        while node is not None:
            # print node.position.board
            if node.child_num() == 0:
                prior_probs = self.rpc.policy_dl_rpc(board_to_stream(node.position.board),
                                                     node.position.get_player_name())
                # normalize prior probs
                prior_probs = normalize_prior_probs(prior_probs)
                node.generate_edges(prior_probs)
            act_q_values = np.empty(node.child_num(), dtype=float)
            for idx, edge in enumerate(node.edges):
                act_q_values[idx] = edge.edge_weight(self.explore_rate)
                # act_q_values[idx] = edge.edge_bonus(self.explore_rate)
            # if node.position.player == RenjuGame.PLAYER_WHITE:  # min for white player
            #     act_q_values = -act_q_values
            best_edge_idx = np.argmax(act_q_values)
            # move to child node
            node_parent = node
            node = node.child[best_edge_idx]
            # store select track
            select_track.append(best_edge_idx)
            print transform_action(node_parent.edges[best_edge_idx].action), ", ",
        print "]"
        # create leaf node
        last_best_edge = node_parent.edges[select_track[-1]]
        leaf_node_position = node_parent.position.replicate_game()
        leaf_node_position.step_games(last_best_edge.action)
        leaf_node = Node(leaf_node_position, parent=node_parent)
        # leaf_node_parent.child[select_track[-1]] = leaf_node
        return leaf_node, select_track

    def expansion(self, leaf_node, select_track):
        """
            expansion phase
        :param leaf_node:
        :param leaf_node_parent:
        :param select_track:
        :return:
        """
        last_select = select_track[-1]
        last_select_edge = leaf_node.parent.edges[last_select]
        if last_select_edge.rollout_rewards > self.visit_threshold:
            print "**expand one node"
            # append leaf node to search tree
            leaf_node.parent.child[last_select] = leaf_node
            # generate edges for new node
            board_stream = board_to_stream(leaf_node.position.board)
            prior_probs = self.rpc.policy_dl_rpc(board_stream, leaf_node.position.get_player_name())
            prior_probs = normalize_prior_probs(prior_probs)
            leaf_node.generate_edges(prior_probs)

    def evaluation(self, leaf_node, random_prob=0.1, host=None):
        """
            evaluation phase
        :param leaf_node:
        :return:
        """
        reward = self.rpc.simulate_rpc("policy_rollout", board_to_stream(leaf_node.position.board),
                                       leaf_node.position.get_player_name(), host=host)
        # game = leaf_node.position.replicate_game()
        # while True:  # loop game
        #     predict_vals = self.rpc.policy_dl_rpc(board_to_stream(game.board), game.get_player_name(), host=host)
        #     # predict_vals = self.rpc.policy_rollout_rpc(board_to_stream(game.board), game.get_player_name())
        #     if random.random() < random_prob:
        #         action = game.choose_action(predict_vals)
        #     else:  # choose second best
        #         action = game.weighted_choose_action(predict_vals)
        #     if action is None:
        #         return 0
        #     _, reward_n, terminal_n = game.step_games(action)
        #     if terminal_n:
        #         return reward_n
        return reward

    def backup(self, root, select_track, out_come):
        """
            backup phase
        :param root:
        :param select_track:
        :param out_come:
        :param leaf_node:
        :return:
        """
        # get value of leaf node
        # leaf_node_value = self.rpc.value_net_rpc(board_to_stream(leaf_node.position.board), leaf_node.position.get_player_name())[0]
        # update in-tree edges
        node = root
        for select in select_track:
            if node.position.player == RenjuGame.PLAYER_BLACK:
                node_out_come = out_come
            else:
                node_out_come = -1 * out_come
            # update selected edge
            edge = node.edges[select]
            edge.rollout_rewards = edge.rollout_rewards - self.virtual_loss + 1.2
            edge.total_rewards = edge.total_rewards + self.virtual_loss + node_out_come
            edge.leaf_evaluate += 0.5
            # edge.total_evaluate += leaf_node_value
            edge.total_evaluate += node_out_come
            edge.update_action_value(self.mix_lambda)
            # move to child
            node = node.child[select]


SIGNAL_FREE = 0
SIGNAL_PAUSE = 1
SIGNAL_RUNNING = 2
SLEEP_INTERVAL = 0.1  # second
MAX_WAIT_TIME = 5 * 60  # minites


class SimulateThread(threading.Thread):
    def __init__(self, pool_manager, name=None):
        threading.Thread.__init__(self, name=name)
        self.signal = SIGNAL_FREE
        self.pool_manager = pool_manager
        self.leaf_node = None
        self.result = None

    def set_signal(self, signal):
        self.signal = signal

    def set_data(self, root):
        self.leaf_node = root

    def free_data(self):
        self.leaf_node = None
        self.result = None

    def run(self):
        while True:
            is_free = self.signal is SIGNAL_FREE
            if is_free:  # pause status or init status
                time.sleep(SLEEP_INTERVAL)
            else:  # running status
                assert (self.leaf_node is not None)
                # evaluation
                host = self.pool_manager.hosts[int(self.name.split("_")[1]) % len(self.pool_manager.hosts)]
                out_come = self.pool_manager.mcts.evaluation(self.leaf_node, host=host)
                self.result = out_come
                self.signal = SIGNAL_FREE


class MCTSThread(threading.Thread):
    def __init__(self, pool_manager, name=None):
        threading.Thread.__init__(self, name=name)
        self.signal = SIGNAL_FREE
        self.setDaemon(True)
        self.root = None
        self.pool_manager = pool_manager
        self.work_time_flag = 0

        self.prev_signal = None
        self.recv_signal = False

    def job_free(self):
        self.set_signal(SIGNAL_FREE)
        self.root = None
        self.prev_signal = None
        self.recv_signal = False

    def job_resume(self, root):
        self.set_signal(SIGNAL_FREE)
        self.root = root

    def set_signal(self, signal):
        if signal is self.signal:
            return
        self.prev_signal = self.signal
        self.signal = signal
        self.recv_signal = False
        while not self.recv_signal:
            time.sleep(SLEEP_INTERVAL)
        if self.signal is SIGNAL_RUNNING:
            self.work_time_flag = time.time()

    def run(self):
        while True:
            is_pause = self.signal is SIGNAL_PAUSE
            is_free = self.signal is SIGNAL_FREE
            if is_pause or is_free or self.root is None:  # pause status or init status
                time.sleep(SLEEP_INTERVAL)
            elif int(time.time() - self.work_time_flag) > MAX_WAIT_TIME:
                logger.warn("too much time in free, release current thread: %s" % self.name)
                self.signal = SIGNAL_FREE
                self.root = None
                self.prev_signal = None
                self.recv_signal = False
            else:  # running status
                # # selection
                # leaf_node, select_track = self.pool_manager.mcts.selection(self.root)
                # # find free simulators
                # simulators = self.pool_manager.get_free_simulators()
                # for _simulate in simulators:
                #     _simulate.set_data(leaf_node)
                #     _simulate.set_signal(SIGNAL_RUNNING)
                # # wait for all simulator finish
                # while True:
                #     loop_lookup_finish = True
                #     time.sleep(SLEEP_INTERVAL)
                #     for _simulate in simulators:
                #         if _simulate.signal is not SIGNAL_FREE:
                #             loop_lookup_finish = False
                #             break
                #     if loop_lookup_finish:
                #         break
                # # merge result of all simulators
                # for _simulate in simulators:
                #     out_come = _simulate.result
                #     # backup
                #     self.pool_manager.mcts.backup(self.root, select_track, out_come)
                #     # expansion
                #     self.pool_manager.mcts.expansion(leaf_node, select_track)
                #     # free data
                #     _simulate.free_data()
                host = self.pool_manager.hosts[int(self.name.split("_")[1]) % len(self.pool_manager.hosts)]
                self.pool_manager.mcts.pipline(self.root, host=host)
                self.pool_manager.mcts.analysis(self.root)
            if self.prev_signal is not self.signal:
                self.recv_signal = True
                self.prev_signal = self.signal


class MCTSThreadPool(object):
    def __init__(self, mcts_model, play_jobs=5, simulate_jobs=5):
        self.hosts = ["http://dpl04.example.com:2223", "http://dpl04.example.com:2227",
                      "http://dpl04.example.com:2228", "http://dpl04.example.com:2229"]
        self.mcts = mcts_model
        self.threads = dict()
        self.simulators = dict()
        for idx in xrange(play_jobs):
            thread = MCTSThread(self, name="play_%d_%d" % (idx, int(time.time())))
            thread.start()
            self.threads[thread.name] = thread
        for idx in xrange(simulate_jobs):
            _simulate = SimulateThread(self, name="simulate_%d_%d" % (idx, int(time.time())))
            _simulate.start()
            self.simulators[_simulate.name] = _simulate

    def decision(self, action, thread_name):
        thread = self.threads[thread_name]
        thread.set_signal(SIGNAL_PAUSE)
        for idx in xrange(thread.root.child_num()):
            if thread.root.edges[idx].action == action:
                if thread.root.child[idx] is None:
                    child_node_position = thread.root.position.replicate_game()
                    child_node_position.step_games(thread.root.edges[idx].action)
                    thread.root.child[idx] = Node(child_node_position, parent=thread.root)
                thread.root = thread.root.child[idx]
                break
        if thread.root.child_num() == 0:
            prior_probs = self.mcts.rpc.policy_dl_rpc(board_to_stream(thread.root.position.board),
                                                      thread.root.position.get_player_name())
            # normalize prior probs
            prior_probs = normalize_prior_probs(prior_probs)
            thread.root.generate_edges(prior_probs)
        thread.root, action = self.mcts.decision(thread.root)
        thread.set_signal(SIGNAL_RUNNING)
        return action

    def simulate(self, thread_name):
        self.threads[thread_name].set_signal(SIGNAL_RUNNING)

    def acquire_thread(self, player):
        for _, _thread in self.threads.items():
            if _thread.signal is SIGNAL_FREE:
                print "ai player:", player
                _thread.root = Node(RenjuGame(board=None, player=player))
                _thread.root.position.board[7][7] = RenjuGame.STONE_BLACK
                if player == "black":
                    _thread.root.position.player = RenjuGame.PLAYER_WHITE
                self.simulate(_thread.name)
                return _thread.name
        return None

    def get_free_simulators(self):
        simulators = []
        for _, _simulate in self.simulators.items():
            if _simulate.signal is SIGNAL_FREE:
                simulators.append(_simulate)
        return simulators

    def free_thread(self, thread_name):
        self.threads[thread_name].job_free()

    def check_auth(self, auth_name):
        if auth_name in self.threads:
            return True
        return False

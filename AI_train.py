#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author:  <yao62995@gmail.com> 

import os
import shutil
import random
import tensorflow as tf

from util.AI_logger import logger
from util.AI_tools import ModelRPC
from AI_import import corpus, patterns, is_legal_stream, stream_to_board
from AI_renju import RenjuGame
from AI_policy_DL import PolicyDLNetwork
from AI_policy_RL import PolicyRLNetwork
from AI_value_net import ValueNetwork
from AI_policy_rollout import PolicyRolloutModel
from AI_mcts import MCTS


def train_policy_rollout(policy_args):
    rollout_features = policy_args.pattern_features
    policy_planes = policy_args.policy_planes
    # init policy dl
    policy_rollout = PolicyRolloutModel(policy_planes, patterns, policy_args,
                                        board_size=policy_args.board_size,
                                        model_dir=policy_args.policy_rollout_models_dir,
                                        gpu=policy_args.policy_rollout_gpu,
                                        optimizer=policy_args.policy_rollout_optimizer,
                                        learn_rate=policy_args.policy_rollout_learn_rate,
                                        distributed_train=False,
                                        )
    # init session
    session = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True))
    # session.run([policy_dl.tf_var["cost"], policy_dl.tf_var["accuracy"], policy_dl.tf_var["optimizer"],
    #              policy_dl.tf_var["out"], policy_dl.tf_var["target"]])
    session.run(tf.initialize_all_variables())
    policy_rollout.set_session(session)
    # restore model if exist
    policy_rollout.restore_model()
    # train network
    policy_rollout.train_policy_rollout(patterns,
                                        epochs=policy_args.policy_rollout_epochs,
                                        batch_size=policy_args.policy_rollout_batch_size)


def train_policy_tree(policy_args):
    pass


def train_policy_network_dl(policy_args):
    policy_planes = policy_args.policy_planes
    # init policy dl
    policy_dl = PolicyDLNetwork(policy_planes, corpus, policy_args, filters=policy_args.policy_dl_filters,
                                board_size=policy_args.board_size,
                                model_dir=policy_args.policy_dl_models_dir, gpu=policy_args.policy_dl_gpu,
                                optimizer=policy_args.policy_dl_optimizer,
                                learn_rate=policy_args.policy_dl_learn_rate,
                                distributed_train=False,
                                )
    # init session
    session = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True))
    # session.run([policy_dl.tf_var["cost"], policy_dl.tf_var["accuracy"], policy_dl.tf_var["optimizer"],
    #              policy_dl.tf_var["out"], policy_dl.tf_var["target"]])
    session.run(tf.initialize_all_variables())
    policy_dl.set_session(session)
    # restore model if exist
    policy_dl.restore_model()

    # train network
    policy_dl.train_policy_network(corpus,
                                   epochs=policy_args.policy_dl_epochs, batch_size=policy_args.policy_dl_batch_size)


def train_policy_network_rl(args):
    policy_planes = args.policy_planes
    # rpc of value_net
    rpc = ModelRPC(args)
    if args.policy_rl_reset:
        # empty old rl policy network
        if os.path.exists(args.policy_rl_models_dir):
            # os.removedirs(args.policy_rl_models_dir)
            shutil.rmtree(args.policy_rl_models_dir)
        os.makedirs(args.policy_rl_models_dir)
        # read parameters from DL policy network
        checkpoint = tf.train.get_checkpoint_state(args.policy_dl_models_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            model_file = checkpoint.model_checkpoint_path
        else:
            logger.error("not found policy dl model avaliable", to_exit=True)
    else:
        model_file = None
    # init policy RL network
    policy_rl = PolicyRLNetwork(policy_planes, args, phase=args.policy_rl_phase, filters=args.policy_rl_filters,
                                board_size=args.board_size,
                                model_dir=args.policy_rl_models_dir, gpu=args.policy_rl_gpu,
                                optimizer=args.policy_rl_optimizer, learn_rate=args.policy_rl_learn_rate,
                                distributed_train=False,
                                )
    # init session
    session = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True))
    session.run(tf.initialize_all_variables())
    policy_rl.set_session(session)
    # restore model if exist
    if model_file is not None:
        policy_rl.saver.restore(session, model_file)
        logger.info("load model file: %s" % model_file)
        policy_rl.save_model("policy_rl", global_step=0)
    else:
        policy_rl.restore_model()
    # train policy rl
    policy_rl.train_policy_network(rpc,
                                   batch_games=args.policy_rl_batch_games,
                                   save_step=args.policy_rl_save_step)


def train_value_network_sl(args):
    value_planes = args.value_planes
    # rpc of policy_dl and policy_rl
    rpc = ModelRPC(args)
    # init value DL network
    value_dl = ValueNetwork(value_planes, args, phase=args.values_net_phase, filters=args.values_net_filters,
                            board_size=args.board_size,
                            model_dir=args.values_net_models_dir, gpu=args.values_net_gpu,
                            optimizer=args.values_net_optimizer, learn_rate=args.values_net_learn_rate,
                            )
    # init session
    session = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True))
    session.run(tf.initialize_all_variables())
    value_dl.set_session(session)
    # restore model if exist
    value_dl.restore_model()
    # train value network
    value_dl.train_value_network(rpc, sample_num=args.values_net_sample_num,
                                 max_time_steps=args.values_net_max_time_steps,
                                 batch_size=args.values_net_batch_size, epochs=args.values_net_epochs)


def load_model(args, model_type, model_file=None):
    policy_planes = args.policy_planes
    value_planes = args.value_planes
    pattern_features = args.pattern_features
    if model_type == "policy_dl":
        model = PolicyDLNetwork(policy_planes, corpus, args, filters=args.policy_dl_filters,
                                board_size=args.board_size,
                                model_dir=args.policy_dl_models_dir, device="gpu", gpu=args.policy_dl_gpu,
                                optimizer=args.policy_dl_optimizer,
                                learn_rate=args.policy_dl_learn_rate,
                                distributed_train=False,
                                )
    elif model_type == "policy_rollout":
        model = PolicyRolloutModel(policy_planes, patterns, args,
                                   board_size=args.board_size,
                                   model_dir=args.policy_rollout_models_dir, device="cpu",
                                   optimizer=args.policy_rollout_optimizer,
                                   learn_rate=args.policy_rollout_learn_rate,
                                   distributed_train=False,
                                   )
    elif model_type == "policy_rl":
        model = PolicyRLNetwork(policy_planes, args, phase=args.policy_rl_phase, filters=args.policy_rl_filters,
                                board_size=args.board_size,
                                model_dir=args.policy_rl_models_dir, device="cpu",
                                optimizer=args.policy_rl_optimizer, learn_rate=args.policy_rl_learn_rate,
                                distributed_train=False,
                                )
    elif model_type == "value_net":
        model = ValueNetwork(value_planes, args, phase=args.values_net_phase, filters=args.values_net_filters,
                             board_size=args.board_size,
                             model_dir=args.values_net_models_dir, device="cpu",
                             optimizer=args.values_net_optimizer, learn_rate=args.values_net_learn_rate,
                             )
    else:
        logger.error("unsupported model type=%s" % model_type, to_exit=True)
    # init session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    session = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                               allow_soft_placement=True,
                                               gpu_options=gpu_options))
    session.run(tf.initialize_all_variables())
    model.set_session(session)
    # restore model
    status = model.restore_model(model_file=model_file)
    if not status and model_type == "policy_rl":
        checkpoint = tf.train.get_checkpoint_state(args.policy_dl_models_dir)
        model_file = checkpoint.model_checkpoint_path
        logger.info("successful load model file: %s" % model_file)
        model.saver.restore(session, model_file)
    return model


def simulate(model_type, model, board, player, random_prob=0.95):
    if player == "black":
        player = RenjuGame.PLAYER_BLACK
    else:
        player = RenjuGame.PLAYER_WHITE
    game = RenjuGame(board=board, player=player)
    while True:  # loop game
        if model_type == "policy_dl" or model_type == "policy_rl":
            state = game.get_states()
            predict_vals = model.predict([state])[0]
        elif model_type == "policy_rollout":
            state = game.get_states(flatten=True)
            predict_vals = model.predict([state])[0]
        elif model_type == "value_net":
            state = game.get_states(player_plane=True)
            predict_vals = model.predict([state])[0]
        if random.random() < random_prob:
            action = game.choose_action(predict_vals)
        else:  # choose second best
            action = game.weighted_choose_action(predict_vals)
        if action is None:
            return 0
        _, reward_n, terminal_n = game.step_games(action)
        if terminal_n:
            return reward_n


def action_model(model_type, model, board, player):
    """
    :param model_type: model type
    :param model: policy model or value model, or else
    :param board: a numpy array with size (15 x 15)
    :param player: a player
    :return:
    """
    if player == "black":
        player = RenjuGame.PLAYER_BLACK
    else:
        player = RenjuGame.PLAYER_WHITE
    position = RenjuGame(board=board, player=player)
    if model_type == "policy_dl" or model_type == "policy_rl":
        state = position.get_states()
        action = model.predict([state])[0]
    elif model_type == "policy_rollout":
        # state = position.get_patterns()
        state = position.get_states(flatten=True)
        action = model.predict([state])[0]
    elif model_type == "value_net":
        state = position.get_states(player_plane=True)
        action = model.predict([state])[0]
    else:
        logger.error("not support model type=%s" % model_type)
        action = None
    return action


def play_games(args):
    player = args.player
    board_stream = args.board
    if board_stream != "":
        if not is_legal_stream(board_stream):
            logger.error("not legal board stream:[%s]" % board_stream, to_exit=True)
        board = stream_to_board(board_stream)
    else:
        board = None
    root = RenjuGame(board=board, player=player)
    rpc = ModelRPC(args)
    mcst = MCTS(rpc, visit_threshold=args.mcts_visit_threshold, virtual_loss=args.mcts_virtual_loss,
                explore_rate=args.mcts_explore_rate, mix_lambda=args.mcts_mix_lambda)
    root = mcst.simulation(root)
    node, action = mcst.decision(root)
    print board
    print "action: %d", action

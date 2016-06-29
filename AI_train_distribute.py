#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author:  <yao62995@gmail.com> 

import os
import shutil
import tensorflow as tf

from util.AI_logger import logger
from AI_import import corpus
from AI_policy_DL import PolicyDLNetwork
from AI_policy_RL import PolicyRLNetwork
from AI_value_net import ValueNetwork


def train_policy_network_dl_distribute(policy_args):
    policy_planes = policy_args.policy_planes
    # hosts
    ps_hosts = policy_args.ps_hosts.split(",")
    worker_hosts = policy_args.worker_hosts.split(",")
    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=policy_args.job_name,
                             task_index=policy_args.task_index)
    if policy_args.job_name == "ps":
        server.join()
    elif policy_args.job_name == "worker":
        # planes = states[0].shape[2]
        # init policy DL network
        policy_dl = PolicyDLNetwork(policy_planes, policy_args, filters=policy_args.policy_dl_filters,
                                    board_size=policy_args.board_size,
                                    model_dir=policy_args.policy_dl_models_dir, gpu=policy_args.policy_dl_gpu,
                                    optimizer=policy_args.policy_dl_optimizer,
                                    learn_rate=policy_args.policy_dl_learn_rate,
                                    distributed_train=True,
                                    )

        init_op = tf.initialize_all_variables()
        summary_op = tf.merge_all_summaries()

        sv = tf.train.Supervisor(is_chief=(policy_args.task_index == 0),
                                 logdir=policy_dl.model_dir,
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=policy_dl.saver,
                                 global_step=policy_dl.global_step,
                                 save_model_secs=0)
        sess = sv.prepare_or_wait_for_session(server.target,
                                              config=tf.ConfigProto(allow_soft_placement=True,
                                                                    log_device_placement=False)
                                              )
        sess.run(init_op)
        # Start queue runners for the input pipelines (if any).
        sv.start_queue_runners(sess)
        # train policy network
        policy_dl.set_session(sess)
        policy_dl.restore_model()
        policy_dl.train_policy_network(corpus,
                                       epochs=policy_args.policy_dl_epochs, batch_size=policy_args.policy_dl_batch_size)


def train_policy_network_rl_distribute(args):
    policy_planes = args.policy_planes
    value_planes = args.value_planes
    # hosts
    ps_hosts = args.ps_hosts.split(",")
    worker_hosts = args.worker_hosts.split(",")
    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=args.job_name,
                             task_index=args.task_index)
    if args.job_name == "ps":
        server.join()
    elif args.job_name == "worker":
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
                                    distributed_train=True,
                                    )
        init_op = tf.initialize_all_variables()
        summary_op = tf.merge_all_summaries()

        sv = tf.train.Supervisor(is_chief=(args.task_index == 0),
                                 logdir=policy_rl.model_dir,
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=policy_rl.saver,
                                 global_step=policy_rl.global_step,
                                 save_model_secs=0)
        sess = sv.prepare_or_wait_for_session(server.target,
                                              config=tf.ConfigProto(allow_soft_placement=True,
                                                                    log_device_placement=True)
                                              )
        sess.run(init_op)
        # Start queue runners for the input pipelines (if any).
        sv.start_queue_runners(sess)
        policy_rl.set_session(sess)
        if model_file is not None:
            policy_rl.saver.restore(sess, model_file)
            logger.info("load model file: %s" % model_file)
        else:
            policy_rl.restore_model()
        # load value network
        if args.policy_rl_phase > 1:
            value_dl = ValueNetwork(value_planes, phase=args.values_net_phase, filters=args.values_net_filters,
                                    board_size=args.board_size,
                                    model_dir=args.values_net_models_dir, gpu=args.values_net_gpu,
                                    optimizer=args.values_net_optimizer, learn_rate=args.values_net_learn_rate,
                                    )
        else:
            value_dl = None
        # train policy rl
        policy_rl.train_policy_network(value_dl, epochs=args.policy_rl_epochs,
                                       batch_games=args.policy_rl_batch_games,
                                       save_step=args.policy_rl_save_step)

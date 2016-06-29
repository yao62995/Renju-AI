#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author:  <yao62995@gmail.com> 

import argparse

from AI_train import *
from util.AI_tools import Utility
from AI_mcts import Node, MCTSThreadPool


def import_corpus_to_db(file_path):
    if os.path.exists(file_path):
        corpus.import_RenjuNet(file_path)


def parser_argument():
    str2bool = lambda v: v.lower() in ("yes", "true", "t", "1")
    parse = argparse.ArgumentParser()
    parse.add_argument("--handle", choices=['train_policy_dl', 'train_policy_rl', 'train_policy_rollout',
                                            'train_value_network', 'play', 'import_corpus'], help="handle type")
    # common parameters
    parse.add_argument("--board_size", type=int, default=15, help="board size")
    parse.add_argument("--policy_planes", type=int, default=3, help="number of policy planes")
    parse.add_argument("--value_planes", type=int, default=4, help="number of value planes")
    parse.add_argument("--pattern_features", type=int, default=64, help="number of pattern features")

    # deploy parameters
    parse.add_argument("--ps_hosts", type=str, default="localhost:2222",
                       help="Comma-separated list of hostname:port pairs")
    parse.add_argument("--worker_hosts", type=str,
                       default="localhost:2223,localhost:2224,localhost:2225,localhost:2226",
                       help="Comma-separated list of hostname:port pairs")
    parse.add_argument("--job_name", type=str, default="worker", help="[ps, worker]")
    parse.add_argument("--task_index", type=int, default=0, help="distributed task index")
    parse.add_argument("--gpu_id", type=int, default=0, help="gpu card id")

    # policy network parameters of supervised learning(DCNN)
    parse.add_argument("--policy_dl_models_dir", type=str, default="./policy_dl_multi_gpu_models", help="policy dl models")
    parse.add_argument("--policy_dl_filters", type=int, default=192, help="policy dl filters")
    parse.add_argument("--policy_dl_optimizer", choices=['rmsprop', 'adam', 'sgd'], default='adam', help='network optimization')
    parse.add_argument("--policy_dl_learn_rate", type=float, default=0.25, help="policy dl learn rate")
    parse.add_argument("--policy_dl_decay_rate", type=float, default=0.5, help="policy dl decay rate")
    parse.add_argument("--policy_dl_epochs_per_decay", type=float, default=20, help="policy dl num epochs per decay")
    parse.add_argument("--policy_dl_gpu", type=int, default=1, help="policy dl train gpu")
    parse.add_argument("--policy_dl_epochs", type=int, default=100, help="policy dl train epochs")
    parse.add_argument("--policy_dl_batch_size", type=int, default=128, help="policy dl train batch size")

    # policy network parameters of supervised learning(rollout)
    parse.add_argument("--policy_rollout_models_dir", type=str, default="./policy_rollout_multi_gpu_models", help="policy rollout models")
    parse.add_argument("--policy_rollout_optimizer", choices=['rmsprop', 'adam', 'sgd'], default='sgd', help='network optimization')
    parse.add_argument("--policy_rollout_learn_rate", type=float, default=0.25, help="policy dl learn rate")
    parse.add_argument("--policy_rollout_decay_rate", type=float, default=0.5, help="policy dl decay rate")
    parse.add_argument("--policy_rollout_epochs_per_decay", type=float, default=50, help="policy dl num epochs per decay")
    parse.add_argument("--policy_rollout_gpu", type=int, default=2, help="policy rollout train gpu")
    parse.add_argument("--policy_rollout_epochs", type=int, default=100, help="policy rollout train epochs")
    parse.add_argument("--policy_rollout_batch_size", type=int, default=128, help="policy rollout train batch size")

    # policy network parameters of reinforcement learning
    parse.add_argument("--policy_rl_models_dir", type=str, default="./policy_rl_multi_gpu_models", help="policy rl models")
    parse.add_argument("--policy_rl_filters", type=int, default=192, help="policy rl filters")
    parse.add_argument("--policy_rl_optimizer", choices=['rmsprop', 'adam', 'sgd'], default='adam', help='network optimization')
    parse.add_argument("--policy_rl_learn_rate", type=float, default=0.00025, help="policy rl learn rate")
    parse.add_argument("--policy_rl_gpu", type=int, default=1, help="policy rl gpu")
    parse.add_argument("--policy_rl_reset", type=str2bool, default=False, help="policy rl model reset")
    parse.add_argument("--policy_rl_phase", type=int, default=1, help="policy rl model phase")
    parse.add_argument("--policy_rl_batch_games", type=int, default=10, help="policy rl train batch games")
    parse.add_argument("--policy_rl_save_step", type=int, default=200, help="policy rl save step")

    # value network parameters
    parse.add_argument("--values_net_models_dir", type=str, default="./value_sl_models", help="values network models")
    parse.add_argument("--values_net_filters", type=int, default=192, help="values network filters")
    parse.add_argument("--values_net_optimizer", choices=['rmsprop', 'adam', 'sgd'], default='sgd', help='network optimization')
    parse.add_argument("--values_net_learn_rate", type=float, default=0.025, help="values network learn rate")
    parse.add_argument("--values_net_gpu", type=int, default=1, help="values network gpu")
    parse.add_argument("--values_net_phase", type=int, default=1, help="values network phase")
    parse.add_argument("--values_net_sample_num", type=int, default=10000, help="values network sampling number")
    parse.add_argument("--values_net_max_time_steps", type=int, default=225, help="values network max time steps")
    parse.add_argument("--values_net_batch_size", type=int, default=128, help="values network batch size")
    parse.add_argument("--values_net_epochs", type=int, default=1, help="values network epochs")

    # MCTS parameters
    parse.add_argument("--mcts_visit_threshold", type=float, default=5, help="mcst visit threshold")
    parse.add_argument("--mcts_virtual_loss", type=float, default=0.2, help="mcst virtual loss")
    parse.add_argument("--mcts_explore_rate", type=float, default=5.0, help="mcst explore rate")
    parse.add_argument("--mcts_mix_lambda", type=float, default=0.75, help="mcst mix lambda")

    # play games
    parse.add_argument("--player", choices=["black", "white"], default="black", help="player role")
    parse.add_argument("--board", type=str, default="", help="board status, format is '010110110100011|0110...'")

    # server parameters
    parse.add_argument("--model_type", choices=["policy_dl", "policy_rl", "policy_rollout", "value_net"],
                       help="ip port for server")
    parse.add_argument("--main_ip_port", type=str, default="localhost:2220", help="ip port for main server")
    parse.add_argument("--policy_dl_ip_port", type=str, default="localhost:2221", help="ip port for policy dl server")
    parse.add_argument("--policy_rl_ip_port", type=str, default="localhost:2222", help="ip port for policy rl server")
    parse.add_argument("--policy_rollout_ip_port", type=str, default="localhost:2223",
                       help="ip port for policy rollout server")
    parse.add_argument("--value_net_ip_port", type=str, default="localhost:2224", help="ip port for value net server")

    # other parameters
    parse.add_argument("--corpus", type=str, default="", help="corpus to import")
    return parse


def run(parse):
    args = parse.parse_args()
    if args.handle == "import_corpus":
        import_corpus_to_db(args.corpus)
    elif args.handle == "train_policy_dl":
        train_policy_network_dl(args)
    elif args.handle == "train_policy_rl":
        train_policy_network_rl(args)
    elif args.handle == "train_policy_rollout":
        train_policy_rollout(args)
    elif args.handle == "train_value_network":
        train_value_network_sl(args)
    elif args.handle == "play":
        play_games(args)
    else:
        parse.print_help()


if __name__ == "__main__":
    arg_parser = parser_argument()
    run(arg_parser)

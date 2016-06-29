#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author:  <yao62995@gmail.com> 

import json
import traceback
from datetime import timedelta
import tensorflow as tf

from flask import Flask, request, jsonify, make_response, request, current_app
from functools import update_wrapper

import Renju as renju


def crossdomain(origin=None, methods=None, headers=None,
                max_age=21600, attach_to_all=True,
                automatic_options=True):
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, basestring):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, basestring):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        def wrapped_function(*args, **kwargs):
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers

            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)

    return decorator


def get_parameter(key, default=None, dtype=None):
    val = request.args.get(key)
    if val is None:
        val = default
    if dtype is not None:
        try:
            val = dtype(val)
        except:
            return default
    return val


def response(**msg):
    return jsonify(msg)


# ref: documentation of Flask, [ http://flask.pocoo.org/docs/0.10/quickstart/ ]
app = Flask(__name__)
args = None
model = None
thread_pool = None
model_dir_dict = None


@app.route('/')
def index():
    return jsonify({'index': "index"})


@app.route('/switch_model', methods=['GET'])
def render_switch_model():
    """
        inner API, not opened
        support params: (model,  "model type, choices=[policy_dl, policy_rl, policy_rollout, value_net]")
                        (file,  "model file path, empty string means most recent checkpoint")
    :return:
    """
    try:
        global model, model_dir_dict
        model_type = get_parameter("model")
        model_file = get_parameter("file", default=None)
        # if model is not None and (model["type"] != model_type or model["file"] != model_file):
        #     # switch model
        #     model["model"].close()
        #     del model
        #     model = None
        #     renju.logger.info("close old model, type=%s, file=%s" % (model["type"], model["file"]))

        model_dir = model_dir_dict[model_type]
        # find avaliable model file
        if model_file is None:
            checkpoint = tf.train.get_checkpoint_state(model_dir)
            if not (checkpoint and checkpoint.model_checkpoint_path):
                renju.logger.warn("switch model error, not found avaliable model file")
                return response(status=2)
            model_file = checkpoint.model_checkpoint_path
            model_file = model_file[model_file.rfind("/") + 1:]
        # check model type
        if model["type"] != model_type:
            renju.logger.warn("switch model error, model type not equal, (%s, %s)" % (model_type, model["type"]))
            return response(status=2)
        if model is None:
            model = {"model": renju.load_model(args, model_type, model_file),
                     "type": model_type,
                     "dir": model_dir + "/",
                     "file": model_file}
        else:
            if model_file != model["file"]:
                model["model"].saver.restore(model["model"].session, model["dir"] + model_file)
                renju.logger.info("successful load model file: %s" % model_file)
        return response(status=0)
    except:
        renju.logger.warn("switch model error, detail=%s" % traceback.format_exc())
        return response(status=2)


@app.route('/action', methods=['POST', 'OPTIONS'])
@crossdomain(origin='*', headers="Origin, X-Requested-With, Content-Type, Accept")
def render_action():
    """
        support params: (model, "model type, choices=[policy_dl, policy_rl, policy_rollout, value_net]")
                        (board,  "board stream")
                        (player, "current player, choices=[black, white]")
    :return:
    """
    try:
        global model
        model_type = get_parameter("model")
        post_data = json.loads(request.data)
        board_stream = post_data["board"].strip()
        player = post_data["player"].strip()
        if model is None or model["type"] != model_type:
            renju.logger.error("model is None or model type not match, please check!")
            return response(status=2)
        else:
            board = renju.stream_to_board(board_stream)
            action = renju.Utility.timeit(lambda: renju.action_model(model_type, model["model"], board, player),
                                          desc="action policy dl, player=%s" % player)
            # desc="action policy dl, board=%s, player=%s" % (board_stream, player))
            if action is not None:
                if model_type != "value_net":
                    return response(status=0, type="decision", action=action.tolist())
                else:
                    return response(status=0, type="decision", value=action[0])
            else:
                return response(status=2)
    except:
        renju.logger.warn("action error, detail=%s" % traceback.format_exc())
        return response(status=2)


@app.route('/simulate', methods=['POST', 'OPTIONS'])
@crossdomain(origin='*', headers="Origin, X-Requested-With, Content-Type, Accept")
def render_simulate():
    """
        support params: (model, "model type, choices=[policy_dl, policy_rl, policy_rollout, value_net]")
                        (board,  "board stream")
                        (player, "current player, choices=[black, white]")
    :return:
    """
    try:
        global model
        post_data = json.loads(request.data)
        board = renju.stream_to_board(post_data["board"].strip())
        player = post_data["player"].strip()
        reward = renju.simulate(model["type"], model["model"], board, player)
        return response(status=0, reward=reward)
    except:
        renju.logger.warn("simulate error, detail=%s" % traceback.format_exc())
        return response(status=2)


@app.route('/play', methods=['POST', 'OPTIONS'])
@crossdomain(origin='*', headers="Origin, X-Requested-With, Content-Type, Accept")
def render_play():
    """
        main API
        support params: (board,  "board stream")
                        (player, "current player, choices=[black, white]")
    :return:
    """
    try:
        global thread_pool
        post_data = json.loads(request.data)
        # board_stream = post_data["board"].strip()
        # player = str(post_data["player"].strip())
        op_action = int(post_data["action"])
        auth_name = str(post_data["auth"]).strip()
        if not thread_pool.check_auth(auth_name):
            return response(status=1, msg="not avaliable auth")
        action = thread_pool.decision(op_action, auth_name)
        return response(status=0, action=action, type="play")
    except:
        renju.logger.warn("play error, detail=%s" % traceback.format_exc())
        return response(status=2)


@app.route('/operate', methods=['GET'])
@crossdomain(origin='*', headers="Origin, X-Requested-With, Content-Type, Accept")
def render_connect():
    """
        play API for connection
        support params: (handle, choices=[connect, release, undo])
                        (player, "current player color, choices=[black, white], optional")
    :return: avaliable thread name
    """
    try:
        global thread_pool
        handle = get_parameter("handle").strip()
        if handle == "connect":
            player = str(get_parameter("player", default="black"))
            auth_name = thread_pool.acquire_thread(player)
            if auth_name is None:
                return response(status=1, msg="no avaliable auth")
            else:
                return response(status=0, msg=auth_name)
        elif handle == "release":
            auth_name = get_parameter("auth")
            thread_pool.free_thread(auth_name)
            return response(status=0, msg="release connect")
        elif handle == "undo":
            auth_name = get_parameter("auth")
            return response(status=0, msg="undo finish")
        else:
            return response(status=1, msg="unknown request handle")
    except:
        renju.logger.warn("game handle error, detail=%s" % traceback.format_exc())
        return response(status=2)


if __name__ == '__main__':
    arg_parser = renju.parser_argument()
    args = arg_parser.parse_args()
    model_dir_dict = {"policy_dl": args.policy_dl_models_dir,
                      "policy_rl": args.policy_rl_models_dir,
                      "policy_rollout": args.policy_rollout_models_dir,
                      "value_net": args.values_net_models_dir}
    # model = renju.load_model(args, args.model_type)
    model_type = args.model_type
    if model_type == "policy_dl":
        ip_port = args.policy_dl_ip_port
    elif model_type == "policy_rl":
        ip_port = args.policy_rl_ip_port
    elif model_type == "policy_rollout":
        ip_port = args.policy_rollout_ip_port
    elif model_type == "value_net":
        ip_port = args.value_net_ip_port
    else:
        ip_port = args.main_ip_port
    host, port = ip_port.split(":")
    if ip_port != args.main_ip_port:
        model = {"model": renju.load_model(args, model_type), "type": model_type,
                 "dir": model_dir_dict[model_type] + "/", "file": None}
    else:
        rpc = renju.ModelRPC(args)
        model = renju.Utility.timeit(
            lambda: renju.MCTS(rpc, visit_threshold=args.mcts_visit_threshold, virtual_loss=args.mcts_virtual_loss,
                               explore_rate=args.mcts_explore_rate, mix_lambda=args.mcts_mix_lambda),
            desc="load MCTS module")
        thread_pool = renju.MCTSThreadPool(model, play_jobs=5, simulate_jobs=3)
    app.run(host=host, port=int(port), debug=False)

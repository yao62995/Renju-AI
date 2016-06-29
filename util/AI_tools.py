#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author:  <yao62995@gmail.com>

import time
import json
import urllib2

from AI_logger import logger


def rpc(url, data=None):
    request_obj = urllib2.Request(url=url, data=data)
    request_obj.add_header('Content-Type', 'application/json; charset=utf-8')
    if data is not None:
        request_obj.add_header('Content-Length', len(data))
    return json.loads(urllib2.urlopen(request_obj).read())


class ModelRPC(object):
    def __init__(self, args):
        self.hosts = dict()
        self.hosts["policy_dl"] = "http://%s" % args.policy_dl_ip_port
        self.hosts["policy_rl"] = "http://%s" % args.policy_rl_ip_port
        self.hosts["policy_rollout"] = "http://%s" % args.policy_rollout_ip_port
        self.hosts["value_net"] = "http://%s" % args.value_net_ip_port

    def switch_model(self, model_type, model_file=None):
        """
        :param model_type: choices in [policy_dl, policy_rl, policy_rollout, value_net]
        :return:
        """
        if model_type not in self.hosts:
            logger.error("not supported model type=%s" % model_type, to_exit=True)
        url = "%s/switch_model?model=%s" % (self.hosts[model_type], model_type)
        if model_file is not None:
            url = "%s&file=%s" % (url, model_file)
        result = rpc(url)
        if result["status"] != 0:
            logger.error("fail to switch model, type=%s, file=%s" % (model_type, model_file), to_exit=True)
        logger.info("success to switch model, type=%s, file=%s" % (model_type, model_file))

    def policy_rpc(self, model_type, board_stream, player, host=None):
        """
        :param model_type: choices in [policy_dl, policy_rl, policy_rollout, value_net]
        :param board_stream:
        :param player:   choices=[black, white]
        :return:
        """
        if host is None:
            host = self.hosts[model_type]
        url = "%s/action?model=%s&debug=1" % (host, model_type)
        data = json.dumps({"board": board_stream, "player": player})
        result = rpc(url, data=data)
        if result["status"] != 0:
            logger.error("fail to request to url = %s" % url)
        return result["action"]

    def simulate_rpc(self, model_type, board_stream, player, host=None):
        """
        :param model_type: choices in [policy_dl, policy_rl, policy_rollout, value_net]
        :param board_stream:
        :param player:   choices=[black, white]
        :return:
        """
        if host is None:
            host = self.hosts[model_type]
        url = "%s/simulate" % host
        data = json.dumps({"board": board_stream, "player": player})
        result = rpc(url, data=data)
        if result["status"] != 0:
            logger.error("fail to request to url = %s" % url)
        return result["reward"]

    def policy_dl_rpc(self, board_stream, player, host=None):
        return self.policy_rpc("policy_dl", board_stream, player, host=host)

    def policy_rl_rpc(self, board_stream, player):
        return self.policy_rpc("policy_rl", board_stream, player)

    def policy_rollout_rpc(self, board_stream, player):
        return self.policy_rpc("policy_rollout", board_stream, player)

    def value_net_rpc(self, board_stream, player):
        return self.policy_rpc("value_net", board_stream, player)


class Utility(object):
    @staticmethod
    def timeit(call_func, desc=None):
        """
            usage: timeit_v2(lambda: func(params))
        :param call_func:
        :return:
        """
        start_time = time.time()
        call_res = call_func()
        elapsed_time = time.time() - start_time
        logger.info('execute=[{}] finished in {} ms'.format(desc, int(elapsed_time * 1000)))
        return call_res

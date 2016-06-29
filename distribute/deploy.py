#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author:  <yao62995@gmail.com>

import argparse

from ssh import SSHConnection
from logger import logger

# CLUSTER_CONFIG = {
#     "ps_hosts": [
#         ("dpl01.example.com:2220", "cpu:0"),
#         ("dpl02.example.com:2220", "cpu:0"),
#         ("dpl03.example.com:2220", "cpu:0"),
#         ("dpl04.example.com:2220", "cpu:0"),
#         ("dpl05.example.com:2220", "cpu:0"),
#         ("dpl01.example.com:2229", "cpu:0"),
#         ("dpl02.example.com:2229", "cpu:0"),
#         ("dpl03.example.com:2229", "cpu:0"),
#         ("dpl04.example.com:2229", "cpu:0"),
#         ("dpl05.example.com:2229", "cpu:0"),
#     ],
#     'worker_hosts': [
#         ("dpl01.example.com:2221", "gpu:0"),
#         ("dpl01.example.com:2222", "gpu:1"),
#         ("dpl01.example.com:2223", "gpu:2"),
#         ("dpl01.example.com:2224", "gpu:3"),
#         ("dpl02.example.com:2221", "gpu:0"),
#         ("dpl02.example.com:2222", "gpu:1"),
#         ("dpl02.example.com:2223", "gpu:2"),
#         ("dpl02.example.com:2224", "gpu:3"),
#         ("dpl03.example.com:2221", "gpu:0"),
#         ("dpl03.example.com:2222", "gpu:1"),
#         ("dpl03.example.com:2223", "gpu:2"),
#         ("dpl03.example.com:2224", "gpu:3"),
#         ("dpl04.example.com:2221", "gpu:0"),
#         ("dpl04.example.com:2222", "gpu:1"),
#         ("dpl04.example.com:2223", "gpu:2"),
#         ("dpl04.example.com:2224", "gpu:3"),
#         ("dpl05.example.com:2221", "gpu:0"),
#         ("dpl05.example.com:2222", "gpu:1"),
#         ("dpl05.example.com:2223", "gpu:2"),
#         ("dpl05.example.com:2224", "gpu:3"),
#     ],
# }

CLUSTER_CONFIG = {
    "ps_hosts": [
        ("dpl01.example.com:2220", "cpu:0"),
        ("dpl02.example.com:2220", "cpu:0"),
        # ("dpl03.example.com:2220", "cpu:0"),
        # ("dpl05.example.com:2220", "cpu:0"),
    ],
    'worker_hosts': [
        ("dpl01.example.com:2221", "gpu:0", "task_0"),
        ("dpl01.example.com:2222", "gpu:1", "task_1"),
        ("dpl01.example.com:2223", "gpu:2", "task_2"),
        ("dpl01.example.com:2224", "gpu:3", "task_3"),
        ("dpl02.example.com:2221", "gpu:0", "task_0"),
        ("dpl02.example.com:2222", "gpu:1", "task_1"),
        ("dpl02.example.com:2223", "gpu:2", "task_2"),
        ("dpl02.example.com:2224", "gpu:3", "task_3"),
        # ("dpl03.example.com:2222", "gpu:0"),
        # ("dpl03.example.com:2223", "gpu:1"),
        # ("dpl03.example.com:2224", "gpu:2"),
        # ("dpl03.example.com:2225", "gpu:3"),
        # ("dpl05.example.com:2222", "gpu:0"),
        # ("dpl05.example.com:2223", "gpu:1"),
        # ("dpl05.example.com:2224", "gpu:2"),
        # ("dpl05.example.com:2225", "gpu:3"),
    ],
}


def get_parameters(kwargs, key, default=None):
    if key in kwargs:
        return kwargs[key]
    else:
        return default


def parse_host_port(host_port):
    """
    :param host_port:  such as "example.com:2222"
    :return: tuple type, (host_name, host_port)
    """
    return tuple(host_port.split(":"))


class ClusterManager(object):
    def __init__(self, config, **kwargs):
        # basic cluster config
        self.config = config

        logger.info("ssh connection")
        user_name = get_parameters(kwargs, "user_name")
        password = get_parameters(kwargs, "password")
        private_key = get_parameters(kwargs, "private_key")
        self.ssh_clients = self.connection(user_name=user_name, password=password, private_key=private_key)

        logger.info("deploy node script, default in user's home directory")
        if user_name != "root":
            default_node_script_path = "/home/%s/node.py" % user_name
        else:
            default_node_script_path = "/root/node.py"
        self.node_script_path = get_parameters(kwargs, "node_script_path", default=default_node_script_path)
        self.deploy_node_script(self.node_script_path)

    def _ps_hosts(self):
        return ','.join(map(lambda _host: _host[0], self.config["ps_hosts"]))

    def _worker_hosts(self):
        return ','.join(map(lambda _host: _host[0], self.config["worker_hosts"]))

    def connection(self, user_name=None, password=None, private_key=None):
        """
            connection target host by ssh protocal
        :param user_name:
        :param password:
        :param private_key:
        :return:
        """
        hosts = set()
        map(lambda _host: hosts.add(parse_host_port(_host[0])[0]), self.config["ps_hosts"])
        map(lambda _host: hosts.add(parse_host_port(_host[0])[0]), self.config["worker_hosts"])
        _ssh_clients = {}
        for host in hosts:
            logger.debug("ssh connect to host[%s]" % host)
            _ssh_clients[host] = SSHConnection(host,
                                               username=user_name,
                                               password=password,
                                               private_key=private_key)
        return _ssh_clients

    def __del__(self):
        map(lambda client: client.close(), self.ssh_clients.values())

    def deploy_node_script(self, script_path):
        # upload script
        for host, _ssh_client in self.ssh_clients.items():
            logger.info("ssh put node script to host[%s], path=[%s]" % (host, script_path))
            _ssh_client.put("node.py", remote_path=script_path)

    def start_node(self, host_port, job_name, task_index):
        logger.info("start node, job=%s, task=%d, host_port=%s" % (job_name, task_index, host_port))
        host, _ = parse_host_port(host_port)
        cmd = "nohup python %s --ps_hosts=%s --worker_hosts=%s --job_name=%s --task_index=%d >/dev/null 2>&1 &" % \
              (self.node_script_path, self._ps_hosts(), self._worker_hosts(), job_name, task_index)
        cmd = "source /etc/profile; " + cmd
        print self.ssh_clients[host].execute(cmd)

    def start_all_node(self):
        map(lambda (idx, _host): self.start_node(_host[0], "ps", idx), enumerate(self.config["ps_hosts"]))
        map(lambda (idx, _host): self.start_node(_host[0], "worker", idx), enumerate(self.config["worker_hosts"]))

    def stop_node(self, host_port, job_name, task_index):
        logger.info("stop node, job=%s, task=%d, host_port=%s" % (job_name, task_index, host_port))
        host, _ = parse_host_port(host_port)
        proc_name = "python %s --ps_hosts=%s --worker_hosts=%s --job_name=%s --task_index=%d" % \
                    (self.node_script_path, self._ps_hosts(), self._worker_hosts(), job_name, task_index)
        cmd = "ps -ef | grep \"%s\" | grep -v grep | awk '{print $2}' | xargs kill" % proc_name
        self.ssh_clients[host].execute(cmd)

    def stop_all_node(self):
        map(lambda (idx, _host): self.stop_node(_host[0], "ps", idx), enumerate(self.config["ps_hosts"]))
        map(lambda (idx, _host): self.stop_node(_host[0], "worker", idx), enumerate(self.config["worker_hosts"]))


def parser_argument():
    cluster = ClusterManager(CLUSTER_CONFIG, user_name="yaojian-xy")
    parse = argparse.ArgumentParser()
    parse.add_argument("--action", choices=['start', 'stop'], help="handle type")
    args = parse.parse_args()
    if args.action == "start":
        cluster.start_all_node()
    elif args.action == "stop":
        cluster.stop_all_node()
    else:
        parse.print_help()


# start the ball rolling.
if __name__ == "__main__":
    parser_argument()

#!/usr/bin/python
#  -*- coding: utf-8 -*-
# author:  <yao62995@gmail.com>

import os
import time
import sys


class Logger(object):
    def __init__(self, log_dir, debug=False):
        self._log_dir = log_dir
        if not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)
        self._debug = debug
        self.DATE_FORMAT = "%Y-%m-%d"
        self.DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
        self._log_date = self._curdate()
        self._logfile = "%s/%s.log" % (self._log_dir, self._log_date)
        self._logger = open(self._logfile, 'a+')

    def _curdate(self):
        return time.strftime(self.DATE_FORMAT, time.localtime())

    def _curdatetime(self):
        return time.strftime(self.DATETIME_FORMAT, time.localtime())

    def _switch_log(self):
        if self._log_date != self._curdate():  # create new logfile
            # close old logfile
            self._logger.close()
            # make new log file
            self._log_date = self._curdate()
            self._logfile = "%s/%s.log" % (self._log_dir, self._log_date)
            self._logger = open(self._logfile, 'a+')

    def _writer(self, msg):
        self._switch_log()
        # maybe locker is needed here
        self._logger.write("%s\n" % msg)
        self._logger.flush()

    def debug(self, msg):
        if self._debug:
            msg = "%s [DEBUG] %s" % (self._curdatetime(), msg)
            self._writer(msg)

    def info(self, msg):
        msg = "%s [INFO] %s" % (self._curdatetime(), msg)
        print msg
        self._writer(msg)

    def warn(self, msg):
        msg = "%s [WARN] %s" % (self._curdatetime(), msg)
        print msg
        self._writer(msg)

    def error(self, msg, to_exit=False):
        msg = "%s [ERROR] %s" % (self._curdatetime(), msg)
        print msg
        self._writer(msg)
        if to_exit:
            sys.exit(-1)


logger = Logger("./log")
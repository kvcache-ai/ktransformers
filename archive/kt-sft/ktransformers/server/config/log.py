#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : unicornchan
Date         : 2024-06-12 02:48:39
Version      : 1.0.0
LastEditors  : chenxl 
LastEditTime : 2024-07-27 01:55:50
'''

import codecs
import logging
import os
import re
import locale
from pathlib import Path
from logging.handlers import BaseRotatingHandler
import time
import colorlog

from ktransformers.server.config.config import Config


class DailyRotatingFileHandler(BaseRotatingHandler):
    """
    such as 'logging.TimeRotatingFileHandler', Additional features:
     - support multiprocess
     - support rotating daily
    """

    def __init__(self, filename, backupCount=0, encoding=None, delay=False, utc=False, **kwargs): # pylint: disable=unused-argument
        self.backup_count = backupCount
        self.utc = utc
        self.suffix = "%Y-%m-%d"
        self.base_log_path = Path(filename)
        if not os.path.exists(self.base_log_path.parent):
            os.makedirs(self.base_log_path.parent)
        self.base_filename = self.base_log_path.name
        self.current_filename = self._compute_fn()
        self.current_log_path = self.base_log_path.with_name(
            self.current_filename)
        BaseRotatingHandler.__init__(self, filename, 'a', encoding, delay)

    # pylint: disable=unused-argument, invalid-name
    def shouldRollover(self, record):
        """
        Determine whether to rotate the log. If the log filename corresponding to the current 
        time is not consistent with the currently opened log filename, then it is necessary
        to rotate the log
        Args:
            record: record is not used, as we are just comparing times, but it is needed so
        the method signatures are the same
        """
        if self.current_filename != self._compute_fn():
            return True
        return False

    def doRollover(self):
        """
        roll over
        """
        # close last log file
        if self.stream:
            self.stream.close()
            self.stream = None  # type: ignore

        # gen new log file name
        self.current_filename = self._compute_fn()
        self.current_log_path = self.base_log_path.with_name(
            self.current_filename)

        if not self.delay:
            self.stream = self._open() # type: ignore

        self.delete_expired_files()

    def _compute_fn(self):
        """
        gen log file name
        """
        return self.base_filename + "." + time.strftime(self.suffix, time.localtime())

    def _open(self):
        """
        open a new log file, create soft link
        """
        if self.encoding is None:
            stream = open(str(self.current_log_path), self.mode, encoding=locale.getpreferredencoding())
        else:
            stream = codecs.open(str(self.current_log_path), self.mode, self.encoding)

        if self.base_log_path.exists():
            try:
                if not self.base_log_path.is_symlink() or os.readlink(self.base_log_path) != self.current_filename:
                    os.remove(self.base_log_path)
            except OSError:
                pass

        try:
            os.symlink(self.current_filename, str(self.base_log_path))
        except OSError:
            pass
        return stream

    def delete_expired_files(self):
        """
        delete expired files every day
        """
        if self.backup_count <= 0:
            return

        file_names = os.listdir(str(self.base_log_path.parent))
        result = []
        prefix = self.base_filename + "."
        plen = len(prefix)
        for file_name in file_names:
            if file_name[:plen] == prefix:
                suffix = file_name[plen:]
                if re.match(r"^\d{4}-\d{2}-\d{2}(\.\w+)?$", suffix):
                    result.append(file_name)
        if len(result) < self.backup_count:
            result = []
        else:
            result.sort()
            result = result[:len(result) - self.backup_count]

        for file_name in result:
            os.remove(str(self.base_log_path.with_name(file_name)))


class Logger(object):
    """
    logger class
    """
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warn': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, level: str = 'info'):
        fmt = '%(asctime)s %(levelname)s %(pathname)s[%(lineno)d] %(funcName)s: %(message)s'
        cfg: Config = Config()
        filename: str = os.path.join(cfg.log_dir, cfg.log_file)
        backup_count: int = cfg.backup_count
        th = DailyRotatingFileHandler(filename=filename, when='MIDNIGHT', backupCount=backup_count, encoding="utf-8")
        th.setFormatter(logging.Formatter(fmt))


        color_fmt = (
            '%(log_color)s%(asctime)s %(levelname)s %(pathname)s[%(lineno)d]: %(message)s'
        )
        color_formatter = colorlog.ColoredFormatter(
            color_fmt,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red'
            }
        )

        sh = logging.StreamHandler()
        sh.setFormatter(color_formatter)

        self.logger = logging.getLogger(filename)
        self.logger.setLevel(self.level_relations.get(level)) # type: ignore
        self.logger.addHandler(th)
        self.logger.addHandler(sh)


logger = Logger(level=Config().log_level).logger

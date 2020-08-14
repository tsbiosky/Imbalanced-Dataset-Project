# -*- coding: utf-8 -*-
# Copyright 2019 Inceptio Technology. All Rights Reserved.
# Author:
#   Jingyu Qian (jingyu.qian@inceptioglobal.ai)

# Python tqdm-compatible colorized logger

import logging

import tqdm


# Color decorators

def highlight_debug(func):
    def wrapper(obj, msg, *args, **kwargs):
        msg = "\033[38;5;94m " + msg + "\033[0m"
        return func(obj, msg, *args, **kwargs)

    return wrapper


def highlight_info(func):
    def wrapper(obj, msg, *args, **kwargs):
        msg = "\033[94m " + msg + "\033[0m"
        return func(obj, msg, *args, **kwargs)

    return wrapper


def highlight_warning(func):
    def wrapper(obj, msg, *args, **kwargs):
        msg = "\033[38;5;214m " + msg + "\033[0m"
        return func(obj, msg, *args, **kwargs)

    return wrapper


def highlight_error(func):
    def wrapper(obj, msg, *args, **kwargs):
        msg = "\033[91m " + msg + "\033[0m"
        return func(obj, msg, *args, **kwargs)

    return wrapper


class TqdmLoggingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)-s - %(message)s'))
        self.setLevel(logging.INFO)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except SystemExit:
            raise SystemExit
        except:
            self.handleError(record)


class TqdmLogger(logging.Logger):
    def __init__(self, name):
        super(TqdmLogger, self).__init__(name)
        self.addHandler(TqdmLoggingHandler())

    @highlight_debug
    def debug(self, msg, *args, **kwargs):
        super(TqdmLogger, self).debug(msg, *args, *kwargs)

    @highlight_info
    def info(self, msg, *args, **kwargs):
        super(TqdmLogger, self).info(msg, *args, **kwargs)

    @highlight_warning
    def warning(self, msg, *args, **kwargs):
        super(TqdmLogger, self).warning(msg, *args, **kwargs)

    @highlight_error
    def error(self, msg, *args, **kwargs):
        super(TqdmLogger, self).error(msg, *args, **kwargs)

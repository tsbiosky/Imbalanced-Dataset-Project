# -*- coding: utf-8 -*-
# Copyright 2019 Inceptio Technology. All Rights Reserved.
# Author:
#   Jingyu Qian (jingyu.qian@inceptioglobal.ai)

# Helper functions to convert values into TensorFlow features that make up
# a tf.train.Example object.

from __future__ import absolute_import, division, print_function
import tensorflow as tf


def is_int_like(obj):
    try:
        return isinstance(obj[0], int)
    except TypeError:
        return isinstance(obj, int)


def is_float_like(obj):
    try:
        return isinstance(obj[0], float)
    except TypeError:
        return isinstance(obj, float)


def is_bytes_like(obj):
    try:
        return hasattr(obj[0], 'decode') or isinstance(obj[0], str)
    except TypeError:
        return hasattr(obj, 'decode') or isinstance(obj, str)


def int64_feature(value):
    """
    Encode int-like features.
    :param value: A list of integers or a single integer.
    :return: tf.train.Feature.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """
    Encode float-like features.
    :param value: A list of float numbers or a single float.
    :return: tf.trian.Feature.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Encode bytes-like feature, such as normal strings or byte strings.
    :param value: A list of strings or bytes, or a single string or a byte.
    :return tf.train.Feature.
    """
    if isinstance(value, list):
        for i in range(len(value)):
            if not isinstance(value[i], bytes):
                value[i] = value[i].encode()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    else:
        if not isinstance(value, bytes):
            value = value.encode()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

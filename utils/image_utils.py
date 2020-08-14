# -*- coding: utf-8 -*-
# Copyright 2019 Inceptio Technology. All Rights Reserved.
# Author:
#   Jingyu Qian (jingyu.qian@inceptioglobal.ai)

# Common image manipulation functions

import cv2
import numpy as np


def resize_image(image: np.ndarray,
                 ratio: float = 0.5,
                 new_width: int = None,
                 new_height: int = None,
                 method='cubic'):
    if method == 'cubic':
        inter = cv2.INTER_CUBIC
    elif method == 'linear':
        inter = cv2.INTER_LINEAR
    elif method == 'nearest':
        inter = cv2.INTER_NEAREST
    else:
        raise ValueError('Unknown interpolation method: {}'.format(method))
    if new_width is None and new_height is None:
        height, width = image.shape[:2]
        new_width = int(width * ratio)
        new_height = int(height * ratio)
    resized = cv2.resize(image, (new_width, new_height), inter)
    return resized

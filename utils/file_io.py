# -*- coding: utf-8 -*-
# Copyright 2019 Inceptio Technology. All Rights Reserved.
# Author:
#   Jingyu Qian (jingyu.qian@inceptioglobal.ai)

# File I/O utilities

import json
import os

import numpy as np
from PIL import Image

from .logging import TqdmLogger

logger = TqdmLogger('file_io')


def load_json(file_path):
    """
    Load JSON file into memory.
    Raises AssertionError if file is not found.
    Args:
        file_path: Path to the JSON file.

    Returns:
        Loaded file contents.
    """
    assert os.path.isfile(file_path), "File {} not found.".format(file_path)
    with open(file_path, 'r') as file:
        file_content = json.loads(file.read())
    return file_content


def dump_json(file_path, content):
    """
    Write content to designated file using JSON.
    Args:
        file_path: Path to the JSON file to write.
        content: Content to be written into the file.

    Returns:
        None
    """
    if not file_path.endswith('.json'):
        file_path = file_path + '.json'
    dirname = os.path.dirname(file_path)
    if dirname:  # Not empty
        os.makedirs(dirname, exist_ok=True)
    with open(file_path, 'w') as file:
        file.write(json.dumps(content))
    logger.info('Dumpped {}.'.format(file_path))


def load_image(file_path, add_batch_dim=False):
    """
    Load an image into memory.
    Args:
        file_path: Path to the image file.
        add_batch_dim: If True, will add an extra batch dimension on 0-th axis.

    Returns:
        An NumPy array of the image and its height and width.
    """
    img_data = np.array(Image.open(file_path))
    height, width = img_data.shape[:2]
    if add_batch_dim:
        return np.expand_dims(img_data, axis=0), height, width
    else:
        return img_data, height, width


def save_image(file_path, content, save_format='jpg'):
    """
    Save an image to designated path.
    Args:
        file_path: Path to the image file to write.
        content: A NumPy array containing image contents to be written to file.
        save_format: Image format. Defaults to 'jpg'.
            Valid values are 'jpg', 'png' and 'bmp'.

    Returns:

    """
    valid_formats = {'jpg', 'png', 'bmp'}
    dirname = os.path.dirname(file_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    img_obj = Image.fromarray(content)
    if file_path.split('.')[-1] not in valid_formats:
        if save_format in valid_formats:
            file_path = '.'.join([file_path, save_format])
        else:
            raise ValueError(
                '\'{}\' is not a valid format. Consider \'jpg\', \'png\' '
                'and \'bmp\'.'.format(save_format))
    img_obj.save(file_path)

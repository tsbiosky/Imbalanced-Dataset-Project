# -*- coding: utf-8 -*-
# Copyright 2019 Inceptio Technology. All Rights Reserved.
# Author:
#   Jingyu Qian (jingyu.qian@inceptioglobal.ai)

# Tool functions to draw various results on to image.

import cv2
import numpy as np
from matplotlib import cm


def _get_color_map(label_texts):
    """
    Prepare a set of rgb values for different object categories.
    The output's length depends on how many different labelts
    there are in the input.
    Args:
        label_texts: A list of labels, each corresponding to a bounding box.

    Returns:
        A dictionary representing RGB values. Each key is a text label, and
        each value is a tupled RGB value.

    """
    if not label_texts or len(label_texts) == 0:
        label_texts = ['default']

    num_classes = len(set(label_texts))
    if num_classes <= 20:
        cmap_name = 'tab20'
    else:
        cmap_name = 'rainbow'
    color_range = np.linspace(0, 1, num_classes)
    rgb_values = cm.get_cmap(cmap_name)(color_range)[:, :3] * 255

    # Explicitly turn values to int, as cv2 drawing function don't support
    # np.int types.
    # BGR order
    rgb_values = [(int(i[2]), int(i[1]), int(i[0])) for i in rgb_values]
    color_map_dictionary = {label: color for label, color in
                            zip(list(set(label_texts)), rgb_values)}
    if 'default' not in color_map_dictionary:
        color_map_dictionary['default'] = rgb_values[0]
    return color_map_dictionary


def draw_bbox_on_img(img, bboxes, label_texts=None, box_coding='xyxy',
                     color_map=None):
    """
    Draw bounding boxes onto image.
    Args:
        img: A NumPy 3-d array of the image to draw on. RGB order.
        bboxes: A list/ndarray of bounding box coordinates. Each box can have
            an additional score value.
        label_texts: A list of bounding box labels. Defaults to None.
        box_coding: One of the following options:
            'xyxy'  - left top right bottom
            'xywh'  - left top width height
            'cxywh' - center_x center_y width height
            Defaults to 'xyxy'.
        color_map: A dictionary conatining RGB values for each label.

    Returns:
        A NumPy 3-d array of the image, with boxes drawn on top.

    """
    img_copy = np.copy(img[:, :, ::-1])  # to BGR
    line_width = 2
    bboxes = np.asarray(bboxes)
    # Each box has 4 coordinates, and 1 optional score
    if not (len(bboxes.shape) == 2 and bboxes.shape[1] in [4, 5]):
        raise ValueError("Invalid box shape: {}".format(bboxes.shape))
    if bboxes.shape[1] == 4:
        c1, c2, c3, c4 = [np.squeeze(i, axis=1) for i in
                          np.split(bboxes, 4, axis=1)]
        c5 = None
    else:
        c1, c2, c3, c4, c5 = [np.squeeze(i, axis=1) for i in
                              np.split(bboxes, 5, axis=1)]
    if box_coding == 'xyxy':
        x1, y1, x2, y2 = (c.astype(np.int) for c in (c1, c2, c3, c4))
    elif box_coding == 'xywh':
        x1, y1, x2, y2 = (c.astype(np.int) for c in (c1, c2, c1 + c3, c2 + c4))
    elif box_coding == 'cxywh':
        x1, y1, x2, y2 = (c.astype(np.int) for c in (
            c1 - 0.5 * c3, c2 - 0.5 * c4, c1 + 0.5 * c3, c2 + 0.5 * c4))
    else:
        raise ValueError('box_coding must be xyxy, xywh or cxywh.')

    # Get color map according to different classes that appear in the image
    if color_map is None:
        color_map = _get_color_map(label_texts)
    for i in range(bboxes.shape[0]):
        # Draw bounding box
        cv2.rectangle(img_copy, (x1[i], y1[i]), (x2[i], y2[i]),
                      color=color_map[label_texts[i]] if label_texts else
                      color_map['default'],
                      thickness=line_width)

        # Draw text and optional score
        if label_texts:
            text = label_texts[i].decode() if \
                isinstance(label_texts[i], bytes) else label_texts[i]
        else:
            text = 'box'

        if c5 is not None:
            s = '{}: {:.2f}'.format(text, c5[i])
        else:
            s = text
        text_size, baseline = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX,
                                              fontScale=0.4, thickness=2)

        pt1 = (x1[i], max(0, y1[i] - text_size[1] - baseline))
        pt2 = (pt1[0] + text_size[0], pt1[1] + text_size[1] + baseline)
        cv2.rectangle(img_copy, pt1, pt2,
                      color_map[label_texts[i]] if label_texts else color_map[
                          'default'], -1)
        cv2.putText(img_copy, s, (pt1[0], pt1[1] + text_size[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)
    return img_copy[:, :, ::-1]


def overlay_mask(img, mask, alpha=0.5):
    """
    Overlay a segmentation mask onto an image using pixel-wise alpha blending.
    Args:
        img: A NumPy 3-d array of the image to draw on.
        mask: A Numpy 2-d or 3-d array with its width/height same as img and
            data type as int, representing pixel-wise classification result.
        alpha: Percatage of value by which the input image will be used.
            Defaults to 0.5.

    Returns:
        A NumPy 3-d array of the blended image.
    """
    if img.shape[:2] != mask.shape[:2]:
        raise ValueError(
            "img W/H {} and mask W/H {} must be the same.".format(
                img.shape[:2], mask.shape[:2]))
    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError(
            "Expected mask dtype to be integer, got {}".format(mask.dtype))

    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=-1)
    if mask.shape[2] == 1:
        mask = np.concatenate([mask] * 3, axis=2)
    classes = np.unique(mask)
    color_map = _get_color_map(list(classes))
    rendered_mask = np.zeros_like(img)
    for key, color in color_map.items():
        if key == 'default':
            continue
        np.putmask(rendered_mask, mask == key, color)
    return np.round(alpha * img + (1 - alpha) * rendered_mask).astype(np.uint8)

def draw_bbox_on_img2(img, bboxes, label_texts=None, box_coding='xyxy',
                     color_map=None):
    """
    Draw bounding boxes onto image.
    Args:
        img: A NumPy 3-d array of the image to draw on. RGB order.
        bboxes: A list/ndarray of bounding box coordinates. Each box can have
            an additional score value.
        label_texts: A list of bounding box labels. Defaults to None.
        box_coding: One of the following options:
            'xyxy'  - left top right bottom
            'xywh'  - left top width height
            'cxywh' - center_x center_y width height
            Defaults to 'xyxy'.
        color_map: A dictionary conatining RGB values for each label.

    Returns:
        A NumPy 3-d array of the image, with boxes drawn on top.

    """
    #img_copy = np.copy(img[:, :, ::-1])  # to BGR
    img_copy=img
    line_width = 2
    bboxes = np.asarray(bboxes)
    # Each box has 4 coordinates, and 1 optional score
    if not (len(bboxes.shape) == 2 and bboxes.shape[1] in [4, 5]):
        raise ValueError("Invalid box shape: {}".format(bboxes.shape))
    if bboxes.shape[1] == 4:
        c1, c2, c3, c4 = [np.squeeze(i, axis=1) for i in
                          np.split(bboxes, 4, axis=1)]
        c5 = None
    else:
        c1, c2, c3, c4, c5 = [np.squeeze(i, axis=1) for i in
                              np.split(bboxes, 5, axis=1)]
    if box_coding == 'xyxy':
        x1, y1, x2, y2 = (c.astype(np.int) for c in (c1, c2, c3, c4))
    elif box_coding == 'xywh':
        x1, y1, x2, y2 = (c.astype(np.int) for c in (c1, c2, c1 + c3, c2 + c4))
    elif box_coding == 'cxywh':
        x1, y1, x2, y2 = (c.astype(np.int) for c in (
            c1 - 0.5 * c3, c2 - 0.5 * c4, c1 + 0.5 * c3, c2 + 0.5 * c4))
    else:
        raise ValueError('box_coding must be xyxy, xywh or cxywh.')

    # Get color map according to different classes that appear in the image
    if color_map is None:
        color_map = _get_color_map(label_texts)
    for i in range(bboxes.shape[0]):
        # Draw bounding box
        cv2.rectangle(img_copy, (x1[i], y1[i]), (x2[i], y2[i]),
                      color=color_map[label_texts[i]] if label_texts else
                      color_map['default'],
                      thickness=line_width)

        # Draw text and optional score
        if label_texts:
            text = label_texts[i].decode() if \
                isinstance(label_texts[i], bytes) else label_texts[i]
        else:
            text = 'box'
        #print(text)
        if c5 is not None:
            s = '{}: {:.2f}'.format(text, c5[i])
        else:
            s = text
        text_size, baseline = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX,
                                              fontScale=0.4, thickness=2)

        pt1 = (x1[i], max(0, y1[i] - text_size[1] - baseline))
        pt2 = (pt1[0] + text_size[0], pt1[1] + text_size[1] + baseline)
        cv2.rectangle(img_copy, pt1, pt2,
                      color_map[label_texts[i]] if label_texts else color_map[
                          'default'], -1)
        cv2.putText(img_copy, s, (pt1[0], pt1[1] + text_size[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)
    return img_copy[:, :, ::-1]
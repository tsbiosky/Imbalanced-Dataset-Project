from enum import Enum, unique

import numpy as np


@unique
class EBox2DCoding(Enum):
    XYXY = 1
    YXYX = 2
    XYWH = 3
    CXYWH = 4


_STR_EBOX2DCODING_MAPPING = {
    'XYXY': EBox2DCoding.XYXY,
    'YXYX': EBox2DCoding.YXYX,
    'XYWH': EBox2DCoding.XYWH,
    'CXYWH': EBox2DCoding.CXYWH
}


class Box2DCoordinateTransformer(object):
    @staticmethod
    def xyxy_to_yxyx(boxes: np.ndarray):
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        return boxes[:, [1, 0, 3, 2]]

    @staticmethod
    def xyxy_to_xywh(boxes):
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        x1, y1, x2, y2 = np.split(boxes, 4, axis=1)
        w, h = x2 - x1, y2 - y1
        return np.concatenate([x1, y1, w, h], axis=1)

    @staticmethod
    def xyxy_to_cxywh(boxes):
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        x1, y1, x2, y2 = np.split(boxes, 4, axis=1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        return np.concatenate([cx, cy, w, h], axis=1)

    @staticmethod
    def yxyx_to_xyxy(boxes):
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        return boxes[:, [1, 0, 3, 2]]

    @staticmethod
    def yxyx_to_xywh(boxes):
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        y1, x1, y2, x2 = np.split(boxes, 4, axis=1)
        w, h = x2 - x1, y2 - y1
        return np.concatenate([x1, y1, w, h], axis=1)

    @staticmethod
    def yxyx_to_cxywh(boxes):
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        y1, x1, y2, x2 = np.split(boxes, 4, axis=1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return np.concatenate([cx, cy, w, h], axis=1)

    @staticmethod
    def xywh_to_xyxy(boxes):
        if boxes.ndims == 1:
            boxes = np.expand_dims(boxes, axis=0)
        x1, y1, w, h = np.split(boxes, 4, axis=1)
        x2, y2 = x1 + w, y1 + h
        return np.concatenate([x1, y1, x2, y2], axis=1)

    @staticmethod
    def xywh_to_yxyx(boxes):
        if boxes.ndims == 1:
            boxes = np.expand_dims(boxes, axis=0)
        x1, y1, w, h = np.split(boxes, 4, axis=1)
        x2, y2 = x1 + w, y1 + h
        return np.concatenate([y1, x1, y2, x2], axis=1)

    @staticmethod
    def xywh_to_cxywh(boxes):
        if boxes.ndims == 1:
            boxes = np.expand_dims(boxes, axis=0)
        x1, y1, w, h = np.split(boxes, 4, axis=1)
        cx, cy = x1 + 0.5 * w, y1 + 0.5 * h
        return np.concatenate([cx, cy, w, h], axis=1)

    @staticmethod
    def cxywh_to_xyxy(boxes):
        if boxes.ndims == 1:
            boxes = np.expand_dims(boxes, axis=0)
        cx, cy, w, h = np.split(boxes, 4, axis=1)
        x1, y1, x2, y2 = cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h
        return np.concatenate([x1, y1, x2, y2], axis=1)

    @staticmethod
    def cxywh_to_yxyx(boxes):
        if boxes.ndims == 1:
            boxes = np.expand_dims(boxes, axis=0)
        cx, cy, w, h = np.split(boxes, 4, axis=1)
        x1, y1, x2, y2 = cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h
        return np.concatenate([y1, x1, y2, x2], axis=1)

    @staticmethod
    def cxywh_to_xywh(boxes):
        if boxes.ndims == 1:
            boxes = np.expand_dims(boxes, axis=0)
        cx, cy, w, h = np.split(boxes, 4, axis=1)
        x1, y1 = cx - 0.5 * w, cy - 0.5 * h
        return np.concatenate([x1, y1, w, h], axis=1)

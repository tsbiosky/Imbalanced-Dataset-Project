# -*- coding: utf-8 -*-
# Copyright 2019 Inceptio Technology. All Rights Reserved.
# Author:
#   Jingyu Qian (jingyu.qian@inceptioglobal.ai)


# This script inspects data written inside a tf-record file
# Usage:
#   python inspect_tfrecord.py tfrecord_file num_of_entries_to_inspect
# Example:
#   python inspect_tfrecord.py /data/train.tfrecord-00000-of-00010 10

import os
import sys
import pickle
import cv2
import numpy as np
import tensorflow as tf
import json
from utils import draw_bbox_on_img, TqdmLogger
#from grab import algo_grabcut
import random
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
logger = TqdmLogger('inspector')




def all_path(dirname):

    result = []#所有的文件

    for maindir, subdir, file_name_list in os.walk(dirname):

        for filename in file_name_list:
            _,type=filename.split('.')
            if type=='tfrecord':
                apath = os.path.join(maindir, filename)#合并成一个完整路径
                result.append(apath)

    return result
t=0
config_file = "./maskrcnn-benchmark/configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"
# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
json_path="./cityscapes/annotations/instancesonly_filtered_gtFine_train.json"
bbox_list={}
with open(json_path) as json_file:
    data = json.load(json_file)
for i in range(len(data['images'])):
            image_path='./cityscapes/images/'+data['images'][i]['file_name']
            img=cv2.imread(image_path)
            _, predictions = coco_demo.run_on_opencv_image(img)
            masks = predictions.get_field("mask").numpy()
            labels = predictions.get_field("labels").data
            boxes = predictions.bbox
            masks = masks.astype(np.uint8)
            num, _, _, _ = masks.shape
            masks = masks.transpose(0, 2, 3, 1)
            for ii in range(num):
                if labels[ii] != 1:
                    continue
                temp = masks[ii]
                big = img * temp
                x1 = int(boxes[ii, 0])
                x2 = int(boxes[ii, 2])
                y1 = int(boxes[ii, 1])
                y2 = int(boxes[ii, 3])
                if y2 - y1 > 256 or x2 - x1 > 256 or y2-y1<120 or x2-x1<50:
                    continue
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                roi = big[y1:y2, x1:x2]
                cv2.imwrite("./bboxes_mask_big/" + str(t) + ".png", roi)
                t += 1
                if t%100==0:
                    print('bboxes:'+str(t))
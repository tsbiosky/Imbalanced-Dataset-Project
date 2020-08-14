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
os.chdir('/mnt/oss/luci-hangzhou/junxuan/notebooks/cvml-kit')
from utils import draw_bbox_on_img, TqdmLogger
#from grab import algo_grabcut
import random
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
logger = TqdmLogger('inspector')
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

def extract_fn(tfrecord):
    image_feature_description = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        #'image/channels': tf.io.FixedLenFeature([], tf.int64),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string)
    }
    sample = tf.io.parse_single_example(tfrecord, image_feature_description)
    image = tf.image.decode_image(sample['image/encoded'])
    img_shape = tf.stack([sample['image/height'],
                          sample['image/width']])
    xmin_ = sample['image/object/bbox/xmin']
    xmax_ = sample['image/object/bbox/xmax']
    ymin_ = sample['image/object/bbox/ymin']
    ymax_ = sample['image/object/bbox/ymax']
    text = sample['image/object/class/text']

    return image, img_shape, xmin_, xmax_, ymin_, ymax_, text

def all_path(dirname):

    result = []#所有的文件

    for maindir, subdir, file_name_list in os.walk(dirname):

        for filename in file_name_list:
            _,type=filename.split('.')
            if type=='tfrecord':
                apath = os.path.join(maindir, filename)#合并成一个完整路径
                result.append(apath)

    return result


def inspect_single_tfrecord( map_fn=extract_fn,
                            num_examples_to_inspect=5):
    """
    Read a single tfrecord file, decodes examples according to map_fn and draw
    bounding boxes.
    Displays a cv2 window and shows decoded images and bounding boxes.
    :param tfrecord_path: Path to the tfrecord file.
    :param map_fn: Mapping function to apply to each example in the tfrecord.
           Used to extract data.
    :param num_examples_to_inspect: Number of images to check.
    :return:
    """
    #tfrecord_path=['/root/tfrecords/img-v001/00/0032df05-210b-4bf6-a882-0aef7961de6e.tfrecord','/root/tfrecords/img-v002/0c/0c66c0cd-33a3-419c-af72-ed2a05875a80.tfrecord']
    #tfrecord_path='/root/tfrecords/img-v001/00/0032df05-210b-4bf6-a882-0aef7961de6e.tfrecord'
    #print(all_path('/root/tfrecords'))
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
    tfrecord_path=all_path('/mnt/oss/luci-hangzhou/junxuan/notebooks/tfrecords')
    assert isinstance(tfrecord_path, list) or isinstance(tfrecord_path, str)
    if isinstance(tfrecord_path, str):
        tfrecord_path = [tfrecord_path]
    tf_dataset = tf.data.TFRecordDataset(tfrecord_path)
    tf_dataset = tf_dataset.map(map_fn)
    max_value = tf.compat.v1.placeholder(tf.int64, shape=[])
    iterator = tf.compat.v1.data.make_initializable_iterator(tf_dataset)
    next_image_data = iterator.get_next()
    #cv2.namedWindow("vis")
    distribution={}
    distribution['center']=[]
    distribution['width']=[]
    distribution['height']=[]
    
    try:
        with tf.compat.v1.Session() as sess:
            sess.run(iterator.initializer,
                     feed_dict={max_value: num_examples_to_inspect})
            #for i in range(num_examples_to_inspect):
            i=1
            t=0 # large bboxe id 
            while(True):
                image_data = sess.run(next_image_data)
                img = image_data[0]
                height, width = image_data[1]
                xmin = image_data[2].values * width
                xmax = image_data[3].values * width
                ymin = image_data[4].values * height
                ymax = image_data[5].values * height
                bboxes = np.stack([xmin, ymin, xmax, ymax], axis=1)
                texts = image_data[6].values.tolist()
                _,predictions = coco_demo.run_on_opencv_image(img)
                masks = predictions.get_field("mask").numpy()
                labels = predictions.get_field("labels").data
                boxes = predictions.bbox
                masks=masks.astype(np.uint8)
                num,_,_,_=masks.shape
                masks=masks.transpose(0,2,3,1)
                for ii in range(num):
                    if labels[ii]!=1:
                        continue
                    temp=masks[ii]
                    big=img*temp
                    x1=int(boxes[ii,0])
                    x2=int(boxes[ii,2])
                    y1=int(boxes[ii,1])
                    y2=int(boxes[ii,3])
                    if y2-y1>256 or x2-x1>256 or y2-y1<120 or x2-x1<50:
                        continue
                    cx=int((x1+x2)/2)
                    cy=int((y1+y2)/2)
                    roi=big[y1:y2,x1:x2]
                    cv2.imwrite("./bboxes_mask_big/"+str(t)+".png", roi)
                    t+=1
                if i%100==0:
                    print('image:'+str(i)+',bboxe:'+str(t))
                #if i==1:
                    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    #img = draw_bbox_on_img(img, bboxes, texts, 'xyxy')
                    #cv2.imwrite("./image/"+str(i)+".png", img)
                    #cv2.imshow("vis", img)
                    #cv2.waitKey(0)
                i+=1
    except Exception as e:
        print(i)
        logger.error(e)
        logger.error(traceback.print_exc())


if __name__ == '__main__':
    logger.info('Invoked as a standalone inspector.')
    inspect_single_tfrecord( num_examples_to_inspect=10)
    logger.info('End of program.')

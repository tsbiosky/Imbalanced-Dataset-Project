import os
import sys
import pickle
import cv2
import numpy as np
import tensorflow as tf
import random
import traceback

from utils import draw_bbox_on_img, TqdmLogger
from datatools.tfrecord import feature
import time
import json
import csv
logger = TqdmLogger('data_aug')

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

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
    label=sample['image/object/class/label']
    format=sample['image/format']
    filename=sample['image/filename']
    encoded=sample['image/encoded']
    return image, img_shape, xmin_, xmax_, ymin_, ymax_,text,label,format,filename,encoded


def all_path(dirname):

    result = []#所有的文件

    for maindir, subdir, file_name_list in os.walk(dirname):

        for filename in file_name_list:
            try:
                _,type=filename.split('.')
                if type=='tfrecord':
                    apath = os.path.join(maindir, filename)#合并成一个完整路径
                    result.append(apath)
            except:
                continue

    return result

def all_path_pickle(dirname):

    result = []#所有的文件

    for maindir, subdir, file_name_list in os.walk(dirname):

        for filename in file_name_list:
                apath = os.path.join(maindir, filename)#合并成一个完整路径
                result.append(apath)
            

    return result
def check(random_x,random_y,h,w,height,width):
    if random_x-w/2<0 or random_x+w/2>width:
        return True
    if random_y-h/2<0 or random_y+h/2>height:
        return True
    return False
def data_aug(map_fn=extract_fn,num_examples_to_inspect=10):
    human=all_path_pickle('./bboxes_human_maskrcnn')
    num_human=len(human)
    dis_path = './distribution_bboxes_human'
    f1 = open(dis_path, 'rb')
    data = pickle.load(f1)
    points=np.array(data['center'])
    x=points[:,0]
    y=points[:,1]
    mu=np.mean((x,y),axis=1)
    con=np.cov(x, y)
    json_path = "./cityscapes/annotations/instancesonly_filtered_gtFine_train.json"
    bbox_list = {}
    cat={}
    with open(json_path) as json_file:
        data = json.load(json_file)
        for i in range(len(data['categories'])):
            name = data['categories'][i]['name']
            id = data['categories'][i]['id']
            cat[id] = name
        for i in range(len(data['images'])):
            bbox_list[data['images'][i]['id']] = []
        for i in range(len(data['annotations'])):
            img_id = data['annotations'][i]['image_id']
            bbox = data['annotations'][i]['bbox']
            x1, y1, w, h = data['annotations'][i]['bbox']
            x1 = int(x1)
            y1 = int(y1)
            w = int(w)
            h = int(h)
            x2 = x1 + w
            y2 = y1 + h
            bbox_list[img_id].append([x1, y1, x2, y2,cat[data['annotations'][i]['category_id']]])

    for img_id in bbox_list.keys():
        image_path='./cityscapes/images/'+data['images'][img_id-500]['file_name']
        img = cv2.imread(image_path)
        add=0
        bboxes=bbox_list[img_id]
        while add<2:
            tryout = 0
            r1 = random.randint(0, num_human - 1)
            path_human = human[r1]
            sample_bbox = cv2.imread(path_human)
            h, w, c = sample_bbox.shape
            sample = np.random.multivariate_normal(mean=mu, cov=con, size=1)
            random_x, random_y = sample[0]
            if check(random_x, random_y, h, w, 1024, 2048):
                continue
            cover = 0
            x1s = int(random_x - w / 2)
            x2s = int(random_x + w / 2)
            y1s = int(random_y - h / 2)
            y2s = int(random_y + h / 2)
            for j in range(len(bboxes)):
                x1 = int(bboxes[j][0])
                x2 = int(bboxes[j][2])
                y1 = int(bboxes[j][1])
                y2 = int(bboxes[j][3])
                img_ppl = img[y1:y2, x1:x2]
                left_column_max = max(x1, x1s)
                right_column_min = min(x2, x2s)
                up_row_max = max(y1, y1s)
                down_row_min = min(y2, y2s)
                if left_column_max >= right_column_min or down_row_min <= up_row_max:
                    cover = 0
                else:
                    cover = 1
                    tryout += 1
                    break
            if cover == 0:
                tryout = 0
            if cover == 1:
                if tryout > 100:
                    print("no space in this image!")
                    break
                continue
            else:
                gray = cv2.cvtColor(sample_bbox, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                notmask = cv2.bitwise_not(mask)
                roi = img[y1s:y2s, x1s:x2s]
                backimage = cv2.bitwise_and(roi, roi, mask=notmask)
                # cv2.imwrite("mask.png", mask)
                # cv2.imwrite("bg.png", backimage)
                # frontpic = cv2.bitwise_and(roi,roi, mask=notmask)
                result = cv2.add(backimage, sample_bbox)
                # roi=cv2.addWeighted(img[y1s:y2s,x1s:x2s],0,sample_bbox,1,0)
                img[y1s:y2s, x1s:x2s] = result
                bbox_list[img_id].append([x1s,y1s,x2s,y2s,'person'])
                add+=1
        cv2.imwrite("./cityscapes_aug/" + str(img_id) + ".png", img)
        if int(img_id)%100==0:
            print(img_id)
    with open('./train_aug.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(data['images'])):
            imgid=i+500
            path='/media/roman/storage/Pedestrian-Synthesis-GAN/cityscapes_aug/'+str(imgid)+'.png'
            for bb in bbox_list[imgid]:
                    writer.writerow([path]+bb)



            
if __name__ == '__main__':
    logger.info('Starting data augmentation')
    data_aug()
    logger.info('End of program.')

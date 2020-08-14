import json
import os
import sys
import pickle
import cv2
import numpy as np
#import tensorflow as tf
import random
import traceback
def all_path(dirname):

    result = []#所有的文件

    for maindir, subdir, file_name_list in os.walk(dirname):

        for filename in file_name_list:
            try:
                _,type=filename.split('.')
                apath = os.path.join(maindir, filename)#合并成一个完整路径
                result.append(apath)
            except:
                continue

    return result
def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
json_path="./cityscapes/annotations/instancesonly_filtered_gtFine_train.json"
bbox_list={}
new_json={}
select=[]
file_path = 'D://ids/key/'
if not os.path.exists(file_path):
    os.makedirs(file_path)
with open(json_path) as json_file:
    data = json.load(json_file)
    for i in range(len(data['images'])):
        bbox_list[data['images'][i]['id']]=[]
    for i in range(len(data['annotations'])):
        if data['annotations'][i]['category_id'] == 1:
            img_id = data['annotations'][i]['image_id']
            bbox=data['annotations'][i]['bbox']
            x1, y1, w, h = data['annotations'][i]['bbox']
            x2 = x1 + w
            y2 = y1 + h
            bbox_list[img_id].append([x1,y1,x2,y2,1])
    for i in range(len(data['images'])):
        if len(bbox_list[data['images'][i]['id']])<2:
            new_json['images'].append(data['images'][i])
            select.append(data['images'][i]['id'])

    for i in range (len(data['annotations'])):
        if data['annotations'][i]['image_id'] in select:
            new_json['annotations'].append(data['annotations'][i])
with open('./data/cityscapes/annotations/instancesonly_filtered_gtFine_train.json', 'w') as outfile:
    json.dump(data, outfile)


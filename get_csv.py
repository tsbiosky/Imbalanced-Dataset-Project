import json
import os
import sys
import pickle
import cv2
import numpy as np
#import tensorflow as tf
import random
import traceback
import csv

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
json_path="./cityscapes/annotations/instancesonly_filtered_gtFine_test.json"
bbox_list={}
cat={}
with open(json_path) as json_file:
    data = json.load(json_file)
    print(data)
    for i in range(len(data['categories'])):
        name = data['categories'][i]['name']
        id = data['categories'][i]['id']
        cat[id]=name
    for i in range(len(data['images'])):
        bbox_list[data['images'][i]['id']]=[]
    for i in range(len(data['annotations'])):
        #if data['annotations'][i]['category_id'] == 1:
        img_id = data['annotations'][i]['image_id']
        bbox=data['annotations'][i]['bbox']
        x1, y1, w, h = data['annotations'][i]['bbox']
        x1=int(x1)
        y1=int(y1)
        w=int(w)
        h=int(h)
        x2 = x1 + w
        y2 = y1 + h
        bbox_list[img_id].append([x1,y1,x2,y2,cat[data['annotations'][i]['category_id']]])
    #for key in data.keys():
        #print(key)
    #print(data['annotations'][0])
    #print(data['categories'][0])
    #print(data['images'][0])
    cnt=1
    #print(data['categories'][2973])
    #for i in range(len(data['images'])):
        #print(data['images'][i]['id'],i)
    cla={}


    with open('./test.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(data['images'])):
            imgid=i
            path='/media/roman/storage/Pedestrian-Synthesis-GAN/cityscapes/images/test/'+data['images'][i]['file_name']
            for bb in bbox_list[imgid]:
                    writer.writerow([path]+bb)
    '''
    with open('./class.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(data['categories'])):
            name=data['categories'][i]['name']
            id=int(data['categories'][i]['id'])-1
            writer.writerow([name,id])
    '''


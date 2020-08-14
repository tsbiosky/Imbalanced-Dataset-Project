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
img_list=all_path("./datasets/images/train1200/")
for i in range(len(img_list)):
    img_path=img_list[i]
    img=cv2.imread(img_path)
    img_name=img_path.split('/')
    img_name=img_name[-1]
    cnt=img_name.split('.')
    cnt=cnt[0]
    ori=img[:,0:256,:]
    json_path="./datasets/bbox/train/" + str(cnt) + ".json"
    data = json.load(open(json_path,"r"))
    #with open("./datasets/bbox/train/" + str(cnt) + ".json","r") as f:
        #data = json.load(f)
    x1=data['x']
    x2=data['w']
    y1=data['y']
    y2=data['h']
    bbox=ori[y1:y2,x1:x2]
    bbox = cv2.cvtColor(bbox, cv2.COLOR_BGR2GRAY)
    bbox=sp_noise(bbox,0.45)
    bbox = cv2.merge([bbox, bbox, bbox])
    noise_img = ori.copy()
    noise_img[y1:y2, x1:x2] = bbox
    img_con = np.concatenate((ori, noise_img), axis=1)
    cv2.imwrite('./datasets/images/train45/' + str(cnt) + '.png',
                img_con)
    #print(cnt)
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
path="./111.png"
img=cv2.imread(path)
img_ppl = img[512-128:512+128, 750-128:750+128]
y1=128-80
y2=128+80
x1=128-25
x2=128+25
bbox = img_ppl[y1:y2, x1:x2]
bbox = cv2.cvtColor(bbox, cv2.COLOR_BGR2GRAY)
bbox = sp_noise(bbox, 0.5)
bbox = cv2.merge([bbox, bbox, bbox])
noise_img = img_ppl.copy()
noise_img[y1:y2, x1:x2] = bbox
img_con = np.concatenate((img_ppl, noise_img), axis=1)
cv2.imwrite('./datasets/images/test/' + str(1) + '.png',
            img_con)
dd = {'x': x1, 'y': y1, 'w': x2, 'h': y2}
with open("./datasets/bbox/test/" + str(1) + ".json",
          "w") as f:
    json.dump(dd, f)

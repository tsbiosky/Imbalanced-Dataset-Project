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
json_path="./cityscapes_ori/annotations/instancesonly_filtered_gtFine_train.json"
bbox_list={}
cnt=0
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
    for i in range(len(data['annotations'])):
        if data['annotations'][i]['category_id']==1 and data['annotations'][i]['iscrowd']==0:
            img_id=data['annotations'][i]['image_id']
            image_path='./cityscapes_ori/images/'+data['images'][img_id-500]['file_name']
            city=data['images'][img_id-500]['file_name'].split('_')
            city=city[0]
            seg_path='./cityscapes_ori/raw/gtFine_trainvaltest/gtFine/train/'+city+'/'+data['images'][img_id-500]['seg_file_name']
            seg=cv2.imread(seg_path)
            x1,y1,w,h=data['annotations'][i]['bbox']
            if h < 70 or w < 25 or h > 256 or w > 256:
                continue
            x2=x1+w
            y2=y1+h
            img=cv2.imread(image_path)
            #cv2.imwrite('./whole.png',img)
            x11 = int((x1 + x2) / 2) - 128
            x22 = int((x1 + x2) / 2) + 128
            dx, dy = 0, 0
            if x11 < 0:
                dx = x11
                x11 = 0
                x22 = 256
            if x22 > 2048:
                dx = x22 - 2048
                x22 = 2048
                x11 = 2048 - 256
            y11 = int((y1 + y2) / 2) - 128
            y22 = int((y1 + y2) / 2) + 128
            if y11 < 0:
                dy = y11
                y11 = 0
                y22 = 256
            if y22 > 1024:
                dy = y22 - 1024
                y22 = 1024
                y11 = 1024 - 256
            #print(cnt)
            img_ppl = img[y11:y22, x11:x22]
            seg_ppl= seg[y11:y22, x11:x22]
            img_ppl = cv2.cvtColor(img_ppl, cv2.COLOR_BGR2RGB)
            ii=np.where(seg_ppl==93)
            seg_ppl=np.zeros((256,256,3))
            seg_ppl[ii]=255
            x1 = 128 - int(w / 2) + dx
            x2 = 128 + int(w / 2) + dx
            y1 = 128 + dy - int(h / 2)
            y2 = 128 + dy + int(h / 2)
            img_bbox=img_ppl[y1:y2,x1:x2]
            seg_bbox=seg_ppl[y1:y2,x1:x2]
            img_ppl=np.zeros((256,256,3))
            img_ppl[y1:y2,x1:x2]=img_bbox
            seg_ppl = np.zeros((256, 256, 3))
            seg_ppl[y1:y2, x1:x2] = seg_bbox
            ii=np.where(seg_ppl==255)
            bg= np.zeros((256, 256, 3))
            bg[ii]=img_ppl[ii]
            cv2.imwrite('./datasets/img/train/'+ str(cnt) + '.png',bg)
            cv2.imwrite('./datasets/seg/train/'+ str(cnt) + '.png',seg_ppl)
            cnt+=1
            print(cnt)

            '''
            cv2.imwrite('./datasets/images/train/' + str(cnt) + '.png',
                        img_con)
            dd = {'x': x1, 'y': y1, 'w': x2, 'h': y2}
            dd['image_path']=image_path
            dd['bbox_list']=bbox_list[img_id]
            with open("./datasets/bbox/train/" + str(cnt) + ".json",
                      "w") as f:
                json.dump(dd, f)
            cnt+=1
            '''
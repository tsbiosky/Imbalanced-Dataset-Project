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
dis_path='./distribution_bboxes_human'
f1 = open(dis_path, 'rb')
data = pickle.load(f1)
points=np.array(data['center'])
x=points[:,0]
y=points[:,1]
mu=np.mean((x,y),axis=1)
con=np.cov(x, y)
f1.close()
with open(json_path) as json_file:
    data = json.load(json_file)
    #for key in data.keys():
        #print(key)
    #print(data['annotations'][0])
    #print(data['categories'][0])
    #print(data['images'][0])

    #print(data['categories'][2973])
    for i in range(len(data['images'])):
        bbox_list[data['images'][i]['id']]=[]
    for i in range(len(data['annotations'])):
        img_id = data['annotations'][i]['image_id']
        bbox=data['annotations'][i]['bbox']
        bbox_list[img_id].append(bbox)
    nn=0
    while(nn<3000):
        r=random.randint(0, len(data['images'])-1)
        img_id=data['images'][r]['id']
        image_path = './cityscapes/images/' + data['images'][r]['file_name']
        img = cv2.imread(image_path)
        sample = np.random.multivariate_normal(mean=mu, cov=con, size=1)
        random_x, random_y = sample[0]
        if random_x<128 or random_x>2048-128 or random_y<128 or random_y>1024-128:
            continue
        x1s = int(random_x - 128)
        x2s = int(random_x + 128)
        y1s = int(random_y - 128)
        y2s = int(random_y + 128)
        cover = 0
        for j in range(len(bbox_list[img_id])):
            x1 = int(bbox_list[img_id][j][0])
            y1 = int(bbox_list[img_id][j][2])
            w = int(bbox_list[img_id][j][1])
            h = int(bbox_list[img_id][j][3])
            x2 = x1 + w
            y2 = y1 + h
            img_ppl = img[y1:y2, x1:x2]
            left_column_max = max(x1, x1s)
            right_column_min = min(x2, x2s)
            up_row_max = max(y1, y1s)
            down_row_min = min(y2, y2s)
            if left_column_max >= right_column_min or down_row_min <= up_row_max:
                cover = 0
            else:
                cover = 1
                break
        if cover==1:
            continue
        if cover==0:
            bg=img[y1s:y2s, x1s:x2s]
            cv2.imwrite('./bg_image/' + str(nn) + '.png',bg)
            print(nn)
            nn+=1
    '''
    for i in range(len(data['images'])):
        







        if data['annotations'][i]['category_id']==1:
            img_id=data['annotations'][i]['image_id']
            image_path='./cityscapes/images/'+data['images'][img_id-500]['file_name']
            x1,y1,w,h=data['annotations'][i]['bbox']
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
            print(cnt)
            img_ppl = img[y11:y22, x11:x22]
            img_ppl = cv2.cvtColor(img_ppl, cv2.COLOR_BGR2RGB)
            x1 = 128 - int(w / 2) + dx
            x2 = 128 + int(w / 2) + dx
            y1 = 128 + dy - int(h / 2)
            y2 = 128 + dy + int(h / 2)
            bbox = img_ppl[y1:y2, x1:x2]
            bbox = cv2.cvtColor(bbox, cv2.COLOR_BGR2GRAY)
            bbox = sp_noise(bbox, 0.5)
            bbox = cv2.merge([bbox, bbox, bbox])
            noise_img = img_ppl.copy()
            noise_img[y1:y2, x1:x2] = bbox
            img_con = np.concatenate((img_ppl, noise_img), axis=1)
            cv2.imwrite('./datasets/images/train/' + str(cnt) + '.png',
                        img_con)
            dd = {'x': x1, 'y': y1, 'w': x2, 'h': y2}
            with open("./datasets/bbox/train/" + str(cnt) + ".json",
                      "w") as f:
                json.dump(dd, f)
            cnt+=1
    '''

import os
import sys
import pickle
import cv2
import numpy as np
import tensorflow as tf
import random
import torch
import traceback
import json
import time
#os.chdir('/mnt/oss/luci-hangzhou/junxuan/notebooks/cvml-kit')
from utils import draw_bbox_on_img, TqdmLogger
from options.test_options import TestOptions
from models.models import create_model
from data.data_loader import CreateDataLoader
from datatools.tfrecord import feature
logger = TqdmLogger('data_aug')
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image



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
    if random_x-128<0 or random_x+128>1536:
        return True
    if random_y-128<0 or random_y+128>1024:
        return True
    return False
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
def pro_data(ab,bbox,path,opt):
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        transform = transforms.Compose(transform_list)

        w_total = opt.loadSize * 2
        w = int(w_total / 2)
        h = opt.loadSize
        w_offset = random.randint(0, max(0, w - opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - opt.fineSize - 1))

        #bbox = json.load(open(bbox_path))
        #bbox = [bbox['y'], bbox['x'], bbox['w'], bbox['h']]
        #print(bbox['y'], bbox['x'], bbox['w'], bbox['h'])
        bbox_x = max(int((bbox['x']/opt.fineSize)*opt.loadSize), 0)
        bbox_y = max(int((bbox['y']/opt.fineSize)*opt.loadSize), 0)
        bbox_w = max(int((bbox['w']/opt.fineSize)*opt.loadSize), 0)
        bbox_h = max(int((bbox['h']/opt.fineSize)*opt.loadSize), 0)

        if bbox_y <= h_offset or bbox_x <= w_offset:
            AB=Image.fromarray(cv2.cvtColor(ab,cv2.COLOR_BGR2RGB))  
            AB = AB.resize((opt.fineSize * 2, opt.fineSize), Image.BICUBIC)
            AB = transform(AB)
            A = AB[:, :opt.fineSize,
               :opt.fineSize]
            B = AB[:, :opt.fineSize,
                opt.fineSize:2*opt.fineSize]
            bbox = [bbox['y'], bbox['x'], bbox['w'], bbox['h']]
        else:
            AB=Image.fromarray(cv2.cvtColor(ab,cv2.COLOR_BGR2RGB))  
            AB = AB.resize((opt.loadSize * 2, opt.loadSize), Image.BICUBIC)
            AB = transform(AB)
            A = AB[:, h_offset:h_offset + opt.fineSize,
               w_offset:w_offset +opt.fineSize]
            B = AB[:, h_offset:h_offset + opt.fineSize,
                w + w_offset:w + w_offset + opt.fineSize]
            bbox = [bbox_y-h_offset, bbox_x-w_offset, bbox_w, bbox_h]
        # print('haha')
        # print(bbox)
        

        if (not opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            #print A.size(2)
            bbox = [bbox[0], A.size(2) - bbox[2], A.size(2) - bbox[1], bbox[3]]
        # print('hehe')
        # print(bbox)
        #print(A.size())
        
        return {'A': A, 'B': B, 'bbox': bbox,
                'A_paths': path, 'B_paths': path}
def get_annotation_from_mask_file(mask_file, scale=1.0):
    '''Given a mask file and scale, return the bounding box annotations

    Args:
        mask_file(string): Path of the mask file
    Returns:
        tuple: Bounding box annotation (xmin, xmax, ymin, ymax)
    '''
    if os.path.exists(mask_file):
        mask = cv2.imread(mask_file)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if len(np.where(rows)[0]) > 0:
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            return int(scale*xmin), int(scale*xmax), int(scale*ymin), int(scale*ymax)
        else:
            return -1, -1, -1, -1
    else:
        print("%s not found. Using empty mask instead."%mask_file)
        return -1, -1, -1, -1
def data_aug():
    json_path = "./cityscapes/annotations/instancesonly_filtered_gtFine_train.json"
    bbox_list = {}
    with open(json_path) as json_file:
        data_city = json.load(json_file)
        for i in range(len(data_city['images'])):
            bbox_list[data_city['images'][i]['id']]=[]
        for i in range(len(data_city['annotations'])):
            img_id = data_city['annotations'][i]['image_id']
            bbox=data_city['annotations'][i]['bbox']
            bbox_list[img_id].append(bbox)
        num_img=len(data_city['images'])
    dis_path = './distribution_bboxes_human'
    f1 = open(dis_path, 'rb')
    data2 = pickle.load(f1)
    points = np.array(data2['center'])
    x = points[:, 0]
    y = points[:, 1]
    mu = np.mean((x, y), axis=1)
    con = np.cov(x, y)
    f1.close()
    nn = 0
    pedestrian = all_path_pickle('./aug_pedestrian')
    num_ped = len(pedestrian)
    with open(json_path) as json_file:
        js = json.load(json_file)
        # img_path=all_path_pickle('./data/cityscapes/leftImg8bit/train')
        # print(img_path)
        tt = len(js['annotations'])
        last_id = js['annotations'][tt - 1]['id']
        # aug_img=all_path_pickle('./possion_blending/img')
        aug_img = all_path_pickle('./aug_img/')
        # aug_mask=all_path_pickle('./possion_blending/mask')
        num_img = len(aug_img)
        total = 1
        for i in range(len(js['images'])):
            # city=str(data['images'][i]['file_name']).split['_'][0]
            file_name = js['images'][i]['file_name']
            img_id = js['images'][i]['id']
            city = file_name.split('_')[0]
            image_path = './cityscapes/leftImg8bit/train/' + js['images'][i]['file_name']
            img = cv2.imread(image_path)
            added=0
            AB = []
            bbox_dict = []
            add_bbox=[]
            tryout = 0
            while (added<6):
                r1 = random.randint(0, num_ped - 1)
                path_pedestrian = pedestrian[r1]
                sample_ped = cv2.imread(path_pedestrian)
                name=path_pedestrian.split('/')
                name=name[-1]
                mask_path='./datasets/seg/test/'+name
                sample_mask=cv2.imread(mask_path)
                x1b, x2b, y1b, y2b = get_annotation_from_mask_file(mask_path)
                w, h, = x2b-x1b, y2b-y1b
                sample = np.random.multivariate_normal(mean=mu, cov=con, size=1)
                random_x, random_y = sample[0]
                random_x=int(random_x)
                random_y=int(random_y)
                if random_x < w or random_x > 2048 - w or random_y < h or random_y > 1024 - h:
                    continue
                x1s = int(random_x - w / 2)
                x2s = int(random_x + w / 2)
                y1s = int(random_y - h / 2)
                y2s = int(random_y + h / 2)
                cover = 0
                for j in range(len(bbox_list[img_id])):
                    x1 = int(bbox_list[img_id][j][0])
                    y1 = int(bbox_list[img_id][j][1])
                    w = int(bbox_list[img_id][j][2])
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
                if cover == 1:
                    tryout += 1
                    if tryout > 200:
                        print("no space in this image!")
                        break
                    continue
                for j in range(len(add_bbox)):
                    x1 = int(add_bbox[j][0])
                    y1 = int(add_bbox[j][1])
                    w = int(add_bbox[j][2])
                    h = int(add_bbox[j][3])
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
                if cover == 1:
                    tryout += 1
                    if tryout > 200:
                        print("no space in this image!")
                        break
                    continue
                if cover == 0:
                    tryout=0
                    sample_ped=cv2.GaussianBlur(sample_ped,(5,5),2)
                    gray = cv2.cvtColor(sample_ped, cv2.COLOR_BGR2GRAY)
                    mask = sample_mask / 255.0
                    # gray = cv2.cvtColor(sample_bbox, cv2.COLOR_BGR2GRAY)
                    # ret, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                    notmask = abs(1 - mask)
                    notmask=notmask[y1b:y2b, x1b:x2b]
                    sample_ped = sample_ped * mask
                    roi = img[y1s:y2s, x1s:x2s]
                    backimage = roi * notmask
                    sample_ped=sample_ped[y1b:y2b, x1b:x2b]
                    #result = cv2.add(backimage, sample_ped)
                    result = backimage+sample_ped
                    # roi=cv2.addWeighted(img[y1s:y2s,x1s:x2s],0,sample_bbox,1,0)
                    img[y1s:y2s, x1s:x2s] = result
                    w, h, = x2s - x1s, y2s - y1s
                    dd = {'x': x1s, 'y': y1s, 'w': w, 'h': h}
                    bbox_dict.append(dd)
                    add_bbox.append([x1b, y1b, h, w])
                    bbox_list[img_id].append([x1b, y1b, h, w])
                    added+=1
                    new_d = {'iscrowd': 0, 'category_id': 24, 'bbox': [x1s, y1s, w, h], 'area': w * h,
                             'segmentation': {'size': [1024, 2048], 'counts': ''}, 'image_id': img_id,
                             'id': str(last_id + total)}
                    js['annotations'].append(new_d)
                    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(image_path, img)
                    #cv2.imwrite('./image_debug_city/tt.png', img)
                    total += 1
                    #if total % 100 == 0:
                    print(total)
                    nn+=len(add_bbox)
    print(nn)
    with open('./cityscapes/annotations/instancesonly_filtered_gtFine_train.json', 'w') as outfile:
        json.dump(js, outfile)
if __name__ == '__main__':
    logger.info('Starting data augmentation')
    data_aug()
    logger.info('End of program.')
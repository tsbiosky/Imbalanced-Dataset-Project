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
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


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
    
def data_aug(map_fn=extract_fn,num_examples_to_inspect=10):
    #human=all_path_pickle('/mnt/oss/luci-hangzhou/junxuan/notebooks/cvml-kit/bboxes_human_mask')
    #num_human=len(human)
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    #print(opt)
    
    model = create_model(opt)
    tfrecord_path=all_path('./tfrecord')
    dis_path='./distribution_bboxes_human'
    f1 = open(dis_path, 'rb')
    data = pickle.load(f1)
    points=np.array(data['center'])
    x=points[:,0]
    y=points[:,1]
    mu=np.mean((x,y),axis=1)
    con=np.cov(x, y)
    f1.close()
    
    for iter_tf in range(0,len(tfrecord_path)):
        tf_path=tfrecord_path[iter_tf]
        tf_path=[tf_path]
        tf_dataset = tf.data.TFRecordDataset(tf_path)
        tf_dataset = tf_dataset.map(map_fn)
        iterator = tf.compat.v1.data.make_initializable_iterator(tf_dataset)
        next_image_data = iterator.get_next()
        ########IO

        #writer = tf.io.TFRecordWriter(new_path)
        ##########
        try:
            with tf.compat.v1.Session() as sess:
                sess.run(iterator.initializer,feed_dict={})
                #for i in range(num_examples_to_inspect):
                i=1
                while(True):
                    image_data = sess.run(next_image_data)
                    img = image_data[0]
                    height, width = image_data[1]
                    xmin = image_data[2].values * width
                    xmax = image_data[3].values * width
                    ymin = image_data[4].values * height
                    ymax = image_data[5].values * height
                    new_xmin=image_data[2].values
                    new_xmax=image_data[3].values
                    new_ymin=image_data[4].values
                    new_ymax=image_data[5].values
                    new_text=image_data[6]
                    new_label=image_data[7].values
                    bboxes = np.stack([xmin, ymin, xmax, ymax], axis=1)
                    texts = image_data[6].values.tolist()
                    str1="person"
                    person=str1.encode(encoding='utf-8').strip()
                    idx=np.argwhere(texts == person)
                    person_bboxes=bboxes[idx]
                    person_bboxes=np.squeeze(person_bboxes)
                    #print(person_bboxes.shape)
                    cur_bboxes=len(idx)
                    tryout=0
                    added=0
                    add_bbox=[]
                    AB=[]
                    bbox_dict=[]
                    while cur_bboxes<6:
                        h,w,=180,90
                        if h>=256 or w>=256:
                            continue
                        sample = np.random.multivariate_normal(mean=mu, cov=con, size=1)
                        random_x,random_y=sample[0]
                        if check(random_x,random_y,h,w,height,width):
                            continue

                        cover=0
                        x1s=int(random_x-w/2)
                        x2s=int(random_x+w/2)
                        y1s=int(random_y-h/2)
                        y2s=int(random_y+h/2)
                        for j in range(len(bboxes)):
                            x1=int(bboxes[j][0])
                            x2=int(bboxes[j][2])
                            y1=int(bboxes[j][1])
                            y2=int(bboxes[j][3])
                            img_ppl=img[y1:y2,x1:x2]
                            left_column_max  = max(x1,x1s)
                            right_column_min = min(x2,x2s)
                            up_row_max       = max(y1,y1s)
                            down_row_min     = min(y2,y2s)
                            if left_column_max>=right_column_min or down_row_min<=up_row_max:
                                cover=0
                            else:
                                cover=1
                                tryout+=1
                                break
                        if cover==0:
                            tryout=0
                        if cover==1:
                            if tryout>100:
                                print("no space in this image!")
                                break
                            continue
                        else:
                            cx=int((x1s+x2s)/2)
                            cy=int((y1s+y2s)/2)
                            roi=img[cy-128:cy+128,cx-128:cx+128]
                            bbox=img[y1s:y2s,x1s:x2s]
                            bbox = cv2.cvtColor(bbox,cv2.COLOR_BGR2GRAY)
                            bbox=sp_noise(bbox,0.5)
                            bbox=cv2.merge([bbox,bbox,bbox])
                            noise_img=roi.copy()
                            noise_img[y1s-cy+128:128+y2s-cy,x1s-cx+128:128-cx+x2s]=bbox
                            #temp=torch.tensor([x1s-cx+128,y1s-cy+128,128-cx+x2s,128+y2s-cy])
                            img_con=np.concatenate((roi, noise_img), axis=1)
                            AB.append(img_con)
                            #cv2.imwrite("./image_debug/"+str(i)+"_input.png", img_con)
                            dd={'x':x1s-cx+128,'y':y1s-cy+128,'w':128-cx+x2s,'h':128+y2s-cy}
                            bbox_dict.append(dd)
                            added+=1
                            add_bbox.append([x1s,y1s,x2s, y2s])
                            sample_bbox = np.stack([x1s, y1s, x2s, y2s], axis=0)
                            sample_bbox=sample_bbox[None]
                            bboxes=np.concatenate((bboxes,sample_bbox),axis = 0)
                            cur_bboxes+=1
                            new_xmin=np.append(new_xmin,float(x1s/width))
                            new_xmax=np.append(new_xmax,float(x2s/width))
                            new_ymin=np.append(new_ymin,float(y1s/height))
                            new_ymax=np.append(new_ymax,float(y2s/height))
                            #person=tf.convert_to_tensor(person)
                            #new_text=np.append(new_text,person)###label,format,filename,encoded
                            #new_text=tf.concat(new_text,person)
                            new_label=np.append(new_label,3)
                            #new_text=new_text.decode('utf-8','ignore')
                    #cv2.imwrite("./image_aug/"+str(i)+".png", img)
                    #print('lol')
                    from data.aligned_dataset2 import AlignedDataset
                    aa = AlignedDataset()
                    aa.initialize(opt,AB,bbox_dict)
                    data_loader = torch.utils.data.DataLoader(
                                    aa,
                                    batch_size=opt.batchSize,
                                    shuffle=not opt.serial_batches,
                                    num_workers=0)
                    #data_loader = CreateDataLoader(opt)
                    dataset = data_loader
                    for ii, data in enumerate(dataset):
                        model.set_input(data)
                        model.test()
                        x1s=add_bbox[ii][0]
                        y1s=add_bbox[ii][1]
                        x2s=add_bbox[ii][2]
                        y2s=add_bbox[ii][3]

                        visuals = model.get_current_visuals()
                        result = cv2.resize(visuals['D2_fake'], (x2s-x1s,y2s-y1s),interpolation=cv2.INTER_CUBIC)
                        img[y1s:y2s,x1s:x2s]=result

                    #print("delate all images!!")
                    '''
                    example = tf.train.Example(features=tf.train.Features(feature={
                    'image/encoded': feature.bytes_feature(cv2.imencode('.png',img)[1].tobytes()),
                    'image/height': feature.int64_feature(height),
                    'image/width': feature.int64_feature(width),
                    'image/object/bbox/xmin': feature.float_feature(new_xmin.tolist()),
                    'image/object/bbox/xmax': feature.float_feature(new_xmax.tolist()),
                    'image/object/bbox/ymin': feature.float_feature(new_ymin.tolist()),
                    'image/object/bbox/ymax': feature.float_feature(new_ymax.tolist()),
                    #'image/object/class/text': feature.bytes_feature(new_text),
                    'image/object/class/label': feature.int64_feature(new_label.tolist()),
                    'image/format': feature.bytes_feature(image_data[8]),
                    'image/filename': feature.bytes_feature(image_data[9])
                    }))
                    writer.write(example.SerializeToString())
                    '''
                    #store new _image 
                    if i%10==0:
                        print('image:'+str(i))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    cv2.imwrite("./image_debug/"+str(i)+"_fake.png", img)
                    i+=1
                    #if i==1:
                    #img = draw_bbox_on_img(img, bboxes, texts, 'xyxy')
                    #cv2.imwrite("./image_debug/"+str(i)+".png", img)
                    #cv2.imshow("vis", img)
                    #cv2.waitKey(0)
                
        except Exception as e:
            print(str(iter_tf)+'_finished_'+'total:'+str(i)+' images')
            #record_tfrecord
            #writer.close()
            #continue
            logger.error(traceback.print_exc())
            
if __name__ == '__main__':
    logger.info('Starting data augmentation')
    data_aug()
    logger.info('End of program.')
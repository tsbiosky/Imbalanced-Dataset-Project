import os
import sys
import pickle
import cv2
import numpy as np
import tensorflow as tf
import random
import traceback
os.chdir('/mnt/oss/luci-hangzhou/junxuan/notebooks/cvml-kit')
from utils import draw_bbox_on_img, TqdmLogger
from datatools.tfrecord import feature
import time
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
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
    human=all_path_pickle('/mnt/oss/luci-hangzhou/junxuan/notebooks/cvml-kit/bboxes_human_maskrcnn')
    num_human=len(human)
    tfrecord_path=all_path('/mnt/oss/luci-hangzhou/junxuan/notebooks/tfrecords')
    dis_path='/mnt/oss/luci-hangzhou/junxuan/notebooks/cvml-kit/distribution_bboxes_human'
    f1 = open(dis_path, 'rb')
    data = pickle.load(f1)
    points=np.array(data['center'])
    x=points[:,0]
    y=points[:,1]
    mu=np.mean((x,y),axis=1)
    con=np.cov(x, y)
    
    for iter_tf in range(0,len(tfrecord_path)):
        tf_path=tfrecord_path[iter_tf]
        tf_path=[tf_path]
        tf_dataset = tf.data.TFRecordDataset(tf_path)
        tf_dataset = tf_dataset.map(map_fn)
        iterator = tf.compat.v1.data.make_initializable_iterator(tf_dataset)
        next_image_data = iterator.get_next()
        ########IO
        path=tf_path[0]
        element=path.split('/')
        element[6]='tfrecords_aug_v5'
        new_path=''
        count=0
        for j in element:
            if count==8:
                count+=1
                continue
            new_path=new_path+'/'+j if count!=0 else new_path+j
            count+=1
        print(new_path)
        writer = tf.io.TFRecordWriter(new_path)
        ##########
        try:
            with tf.compat.v1.Session() as sess:
                sess.run(iterator.initializer,feed_dict={})
                #for i in range(num_examples_to_inspect):
                i=1
                while(True):
                    time_start=time.time()
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
                    str2="car"
                    car=str2.encode(encoding='utf-8').strip()
                    str3="truck"
                    truck=str3.encode(encoding='utf-8').strip()
                    str4="bus"
                    bus=str4.encode(encoding='utf-8').strip()
                    str5="bicycle"
                    bicycle=str5.encode(encoding='utf-8').strip()
                    texts=np.array(texts)
                    idx_5=np.argwhere((texts == person)|(texts==car)|(texts==truck)|(texts==bus)|(texts==bicycle))
                    idx=np.argwhere(texts == person)
                    idc=np.argwhere(texts == car)
                    person_bboxes=bboxes[idx]
                    fix_num=max(4,2*len(idc))
                    void_bboxes=bboxes[idx_5]
                    void_bboxes=np.squeeze(void_bboxes)
                    person_bboxes=np.squeeze(person_bboxes)
                    #print(person_bboxes.shape)
                    cur_bboxes=len(idx)
                    tryout=0
                    while cur_bboxes<fix_num:
                        r1=random.randint(0,num_human-1)
                        path_human=human[r1]
                        sample_bbox = cv2.imread(path_human)
                        h,w,c=sample_bbox.shape
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
                            gray = cv2.cvtColor(sample_bbox, cv2.COLOR_BGR2GRAY)
                            ret, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                            notmask = cv2.bitwise_not(mask)
                            roi=img[y1s:y2s,x1s:x2s]
                            backimage = cv2.bitwise_and(roi, roi, mask=notmask)
                            #cv2.imwrite("mask.png", mask)
                            #cv2.imwrite("bg.png", backimage)
                            #frontpic = cv2.bitwise_and(roi,roi, mask=notmask)
                            result = cv2.add(backimage, sample_bbox)
                            #roi=cv2.addWeighted(img[y1s:y2s,x1s:x2s],0,sample_bbox,1,0)
                            img[y1s:y2s,x1s:x2s]=result
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

                    #store new _image 
                    if i%100==0:
                        print('image:'+str(i))
                    i+=1
                    #time_end=time.time()
                    #print('time_cost:',time_end-time_start,'s')
                    #if i==1:
                    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    #img = draw_bbox_on_img(img, bboxes, texts, 'xyxy')
                    #cv2.imwrite("./image_debug/"+str(i)+".png", img)
                    #print(i)
                    #cv2.imshow("vis", img)
                    #cv2.waitKey(0)
                
        except Exception as e:
            print(str(iter_tf)+'_finished_'+'total:'+str(i)+' images')
            #record_tfrecord
            writer.close()
            continue
            logger.error(traceback.print_exc())
            
if __name__ == '__main__':
    logger.info('Starting data augmentation')
    data_aug()
    logger.info('End of program.')
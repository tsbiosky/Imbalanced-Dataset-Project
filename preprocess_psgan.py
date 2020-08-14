import os
import sys
import pickle
import cv2
import numpy as np
#import tensorflow as tf
import random
import traceback
import json
#tf.compat.v1.enable_eager_execution()

#os.chdir('/mnt/oss/luci-hangzhou/junxuan/notebooks/cvml-kit')
from utils import draw_bbox_on_img, TqdmLogger
from datatools.tfrecord import feature

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
def data_aug(map_fn=extract_fn,num_examples_to_inspect=10):
    human=all_path_pickle('/mnt/oss/luci-hangzhou/junxuan/notebooks/cvml-kit/bboxes_human_mask')
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
    
    for iter_tf in range(2,len(tfrecord_path)):
        tf_path=tfrecord_path[iter_tf]
        tf_path=[tf_path]
        tf_dataset = tf.data.TFRecordDataset(tf_path)
        tf_dataset = tf_dataset.map(map_fn)
        iterator = tf.compat.v1.data.make_initializable_iterator(tf_dataset)
        next_image_data = iterator.get_next()
        ########IO
        path=tf_path[0]
        element=path.split('/')
        element[6]='tfrecords_aug_v3'
        new_path=''
        count=0
        for j in element:
            if count==8:
                count+=1
                continue
            new_path=new_path+'/'+j if count!=0 else new_path+j
            count+=1
        #print(new_path)
        writer = tf.io.TFRecordWriter(new_path)
        ##########
        try:
            with tf.compat.v1.Session() as sess:
                sess.run(iterator.initializer,feed_dict={})
                #for i in range(num_examples_to_inspect):
                i=1
                t=0
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
                    texts=np.array(texts)
                   
                    idx=np.argwhere(texts == person)
                    person_bboxes=bboxes[idx]
                    person_bboxes=np.squeeze(person_bboxes)
                    #print(person_bboxes.shape)
                    cur_bboxes=len(idx)
                    for j in range(len(idx)):
                        x1=int(person_bboxes[j][0])
                        x2=int(person_bboxes[j][2])
                        y1=int(person_bboxes[j][1])
                        y2=int(person_bboxes[j][3])
                        width=abs(x2-x1)
                        height=abs(y2-y1)
                        if height<70 or width<25 or height >256 or width>256:
                            continue
                        x11=int((x1+x2)/2)-128
                        x22=int((x1+x2)/2)+128
                        dx,dy=0,0
                        if x11<0:
                            dx=x11
                            x11=0
                            x22=256
                        if x22>1535:
                            dx=x22-1535
                            x22=1536
                            x11=1536-256
                        y11=int((y1+y2)/2)-128
                        y22=int((y1+y2)/2)+128
                        if y11<0:
                            dy=y11
                            y11=0
                            y22=256
                        if y22>1024:
                            dy=y22-1024
                            y22=1024
                            y11=1024-256
                        print(y11,y22,x11,x22)
                        img_ppl=img[y11:y22,x11:x22]
                        img_ppl = cv2.cvtColor(img_ppl, cv2.COLOR_BGR2RGB)
                        x1=128-int(width/2)+dx
                        x2=128+int(width/2)+dx
                        y1=128+dy-int(height/2)
                        y2=128+dy+int(height/2)
                        bbox=img_ppl[y1:y2,x1:x2]
                        bbox = cv2.cvtColor(bbox,cv2.COLOR_BGR2GRAY)
                        bbox=sp_noise(bbox,0.5)
                        bbox=cv2.merge([bbox,bbox,bbox])
                        noise_img=img_ppl.copy()
                        noise_img[y1:y2,x1:x2]=bbox
                        img_con=np.concatenate((img_ppl, noise_img), axis=1)
                        cv2.imwrite('/mnt/oss/luci-hangzhou/junxuan/notebooks/psgan/dataset/images/train/'+str(t)+'.png',img_con)
                        dd={'x':x1,'y':y1,'w':x2,'h':y2}
                        with open("/mnt/oss/luci-hangzhou/junxuan/notebooks/psgan/dataset/bbox/train/"+str(t)+".json","w") as f:
                            json.dump(dd,f)
                        #cv2.imwrite("./image_debug/"+str(t)+".png", img_ppl)
                        #cv2.imwrite("./image_debug/"+str(t)+"_bbox.png", img)
                        t+=1
                    

                    #store new _image 
                    if i%100==0:
                        print('image:'+str(i))
                    i+=1
                    #if i==1:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    #img = draw_bbox_on_img(img, bboxes, texts, 'xyxy')
                    #cv2.imwrite("./image_debug/"+str(i)+".png", img)
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
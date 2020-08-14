import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import json
import random
import cv2
import numpy as np
import skimage.io
import skimage.transform
import skimage.color
import skimage
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
class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        if opt.phase!='test':
            self.dir_img = os.path.join(opt.dataroot, 'img', opt.phase)
            self.img_path = all_path(self.dir_img)
            self.img_path = sorted(self.img_path)
        self.dir_seg = os.path.join(opt.dataroot, 'seg', opt.phase)
        self.seg_path = all_path(self.dir_seg)
        self.seg_path = sorted(self.seg_path)
        #self.AB_paths, self.bbox_paths = sorted(make_dataset(self.dir_AB, self.dir_bbox))
        #self.AB_paths, self.bbox_paths = make_dataset(self.dir_AB, self.dir_bbox)



        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def load_image(self, image_path):
        img = skimage.io.imread(image_path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0
    def __getitem__(self, index):

        #print(AB_path)
        seg_path = self.seg_path[index]
        seg = self.load_image(seg_path)
        if self.opt.phase!='test':
            img_path = self.img_path[index]
            img=self.load_image(img_path)
            return {'img': img, 'seg': seg ,'seg_path':seg_path}
        return{'seg': seg,'seg_path':seg_path}
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



    def __len__(self):
        return len(self.seg_path)

    def name(self):
        return 'AlignedDataset'

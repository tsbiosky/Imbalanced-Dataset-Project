import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import json


class AlignedDataset(BaseDataset):
    def initialize(self, opt,AB,bbox):
        self.opt = opt
        self.AB=AB
        self.bbox=bbox

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        AB = self.AB[index]
        #print(AB_path)
        bbox= self.bbox[index]
        #print(bbox_path,"sss")
        path='./temp'
        w_total = self.opt.loadSize * 2
        w = int(w_total / 2)
        h = self.opt.loadSize
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        
        #bbox = [bbox['y'], bbox['x'], bbox['w'], bbox['h']]
        #print(bbox['y'], bbox['x'], bbox['w'], bbox['h'])
        bbox_x = max(int((bbox['x']/self.opt.fineSize)*self.opt.loadSize), 0)
        bbox_y = max(int((bbox['y']/self.opt.fineSize)*self.opt.loadSize), 0)
        bbox_w = max(int((bbox['w']/self.opt.fineSize)*self.opt.loadSize), 0)
        bbox_h = max(int((bbox['h']/self.opt.fineSize)*self.opt.loadSize), 0)

        if bbox_y <= h_offset or bbox_x <= w_offset:
            #AB = Image.open(AB_path).convert('RGB')
            AB = Image.fromarray(AB, mode='RGB')
            AB = AB.resize((self.opt.fineSize * 2, self.opt.fineSize), Image.BICUBIC)
            AB = self.transform(AB)
            A = AB[:, :self.opt.fineSize,
               :self.opt.fineSize]
            B = AB[:, :self.opt.fineSize,
                self.opt.fineSize:2*self.opt.fineSize]
            bbox = [bbox['y'], bbox['x'], bbox['w'], bbox['h']]
        else:
            #AB = Image.open(AB_path).convert('RGB')
            AB = Image.fromarray(AB, mode='RGB')
            AB = AB.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
            AB = self.transform(AB)
            A = AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
            B = AB[:, h_offset:h_offset + self.opt.fineSize,
                w + w_offset:w + w_offset + self.opt.fineSize]
            bbox = [bbox_y-h_offset, bbox_x-w_offset, bbox_w, bbox_h]
        # print('haha')
        # print(bbox)
        

        if (not self.opt.no_flip) and random.random() < 0.5:
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

    def __len__(self):
        return len(self.AB)

    def name(self):
        return 'ffff'

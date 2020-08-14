import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
import cv2
opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
# test
for i, data in enumerate(dataset):
#    if i >= opt.how_many:
#        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    seg_path=model.get_image_paths()
    print('process image... %s' % str(i))
    seg_path=seg_path[0].split('/')
    seg_name=seg_path[-1]
    cv2.imwrite('./aug_pedestrian/'+seg_name,visuals['fake'])
    #cv2.imwrite('./'+str(i)+'real_b.png',visuals['real_A'])
    #visualizer.save_images(webpage, visuals, img_path)


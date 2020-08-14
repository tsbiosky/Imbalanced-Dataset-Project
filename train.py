import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import cv2
import os
import numpy as np
from util.visualizer import Visualizer

opt = TrainOptions().parse()
opt.filename='train45'
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)
#print(opt)
model = create_model(opt)
opt.isTrain=True
#visualizer = Visualizer(opt)

CRITIC_ITERS = 5
total_steps = 0
iter_d = 0
only_d = False

for epoch in range(1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    for i, data in enumerate(dataset):
        #print("image:"+str(i))
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - dataset_size * (epoch - 1)
        model.set_input(data)
        if iter_d <= CRITIC_ITERS-1:
           only_d = False
        else:
           only_d = True
        try:
            model.optimize_parameters(only_d)
        except:
            print("skip image:" )
            continue

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')
        iter_d += 1
        if iter_d == 6:
            iter_d = 0

    try:
        errors = model.get_current_errors()
    except:
        print("skip image:")
        continue
    t = (time.time() - iter_start_time) / opt.batchSize
    print(epoch,errors)
    #if epoch % opt.save_epoch_freq == 0 or epoch==1:
    print('saving the model at the end of epoch %d, iters %d' %
          (epoch, total_steps))
    model.save('latest')
    model.save(epoch)
    re=model.get_current_visuals()
    img=re['img']
    seg=re['seg']
    fake=re['fake']
    dis=np.concatenate((img,seg,fake),axis = 1)
    cv2.imwrite("./image_debug/" +str(epoch) + ".png",dis)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.niter:
        model.update_learning_rate()

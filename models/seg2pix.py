import numpy as np
import torch
import os
import cv2
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from copy import deepcopy
from . import networks
from PIL import Image
import torchvision.transforms as transforms
from retinanet import model
use_cuda=True
class seg2pix(BaseModel):
    def name(self):
        return 'seg2pix'


    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.opt = opt
        self.isTrain = opt.isTrain

        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids)
        #self.netG = networks.define_G(3, 3,opt.ngf, "PATN", "instance", not True, "normal",
                                      #0,n_downsampling=2)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_image = networks.define_image_D( opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
            use_sigmoid = not opt.no_lsgan

        if not self.isTrain or opt.continue_train:
            #print(opt.which_epoch)
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD_image, 'D_image', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            #print('haha'+ str(opt.no_lsgan))
            # self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionGAN_image = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_image = torch.optim.Adam(self.netD_image.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD_image)
        print('-----------------------------------------------')
    def calc_gradient_penalty(netD, real_data, fake_data):
        # print real_data.size()
        alpha = torch.rand(BATCH_SIZE, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda() if use_cuda else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        if use_cuda:
            interpolates = interpolates.cuda()
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(
                                      ) if use_cuda else torch.ones(
                                      disc_interpolates.size()),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty
    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.seg = (input['seg'])
        self.seg_path=input['seg_path']
        if self.isTrain:
            self.img = (input['img'])
            self.img = self.img.transpose(1, 3).float().cuda()
        #print(self.seg.numpy().shape,self.img.numpy().shape)
        #cv2.imwrite("./image_debug/ img.png", self.img.squeeze().numpy()*255.0)
        #cv2.imwrite("./image_debug/ seg.png", self.seg.squeeze().numpy()*255.0)
        self.seg=self.seg.transpose(1,3).float().cuda()

                #print(input_A.size())

    def forward(self):

        self.fake = self.netG.forward(self.seg)

    # no backprop gradients
    def test(self):

        self.fake = self.netG.forward(self.seg)

    # get image paths
    def get_image_paths(self):
        return self.seg_path

    def backward_D_image(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        self.pred_fake = self.netD_image.forward(self.fake)
        # self.loss_D_image_fake = self.criterionGAN(self.pred_fake, False)
        self.loss_D_image_fake = self.criterionGAN_image(self.pred_fake, False)
        #self.loss_D_image_fake = torch.nn.ReLU()(1.0 + self.pred_fake).mean()
        # Real
        self.pred_real = self.netD_image.forward(self.img)
        # self.loss_D_image_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_image_real = self.criterionGAN_image(self.pred_real, True)
        #self.loss_D_image_real= torch.nn.ReLU()(1.0 - self.pred_real).mean()
        # Combined loss
        self.loss_D_image = (self.loss_D_image_fake + self.loss_D_image_real) * 0.5

        self.loss_D_image.backward()



    def backward_G(self):
        # First, G(A) should fake the discriminator1 and discriminator1
        # discriminator1

        pred_fake_image = self.netD_image.forward(self.fake)
        # self.loss_G_GAN_image = self.criterionGAN(pred_fake_image, True)
        self.loss_G_GAN_image = self.criterionGAN_image(pred_fake_image, True)
        #self.loss_G_GAN_image = - pred_fake_image.mean()
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake, self.img) * self.opt.lambda_A


        self.loss_G = self.loss_G_GAN_image + self.loss_G_L1

        self.loss_G.backward()




    def optimize_parameters(self, only_d):

        self.forward()
        self.optimizer_D_image.zero_grad()
        self.backward_D_image()
        self.optimizer_D_image.step()
        #self.backward_det()
        #self.optimizer_det.step()
        
        self.forward()
        
        if only_d == False:
            self.forward()
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN_image', self.loss_G_GAN_image.data),
                            ('G_L1', self.loss_G_L1.data),
                            ('D_image', self.loss_D_image.data)
                            #('det_loss_real',self.det_real.data),
                            #('det_loss_fake', self.det_fake.data)
                            ])

    def get_current_visuals(self):

        fake=self.fake.squeeze().transpose(0,2).cpu().float().detach().numpy()*255.0
        seg=self.seg.squeeze().transpose(0,2).cpu().float().detach().numpy()*255.0
        if self.isTrain:
            img=self.img.squeeze().transpose(0,2).cpu().float().detach().numpy()*255.0
            return OrderedDict([('fake', fake), ('seg', seg), ('img', img)])
        return OrderedDict([('fake', fake), ('seg', seg)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD_image, 'D_image', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D_image.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

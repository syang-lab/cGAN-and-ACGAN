#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 23:39:16 2021

@author: shanyang
"""

import torch
import torch.nn as nn
import torch.optim as optim
from gen_dec import ACDiscriminator
from gen_dec import ACGenerator
from train import acgan_training
from train import cgan_training
from train import initialize_weights
from train import acgan_calculate_IS
from train import cal_ms_ssim

import torchvision
import torchvision.transforms as transforms
from torchvision.models.inception import inception_v3
import torchvision.utils as vutils

import numpy as np



if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    nz=100
    nclass=10
    
    lr=0.0002
    iterations=1000
    
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    batch_size = 100
    trainset = torchvision.datasets.CIFAR10(root='./data', download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    
    dis_model = ACDiscriminator(nclass).to(device)
    gen_model = ACGenerator(nz).apply(initialize_weights).to(device)
    
    label_loss=nn.BCELoss()
    class_loss=nn.NLLLoss()
    
    optD = optim.Adam(dis_model.parameters(), lr=lr)
    optG = optim.Adam(gen_model.parameters(), lr=lr)
    
    
    acgan_training(iterations, trainloader, dis_model, gen_model, optD, optG, class_loss, label_loss, nz, device)
    
    inception_model = inception_v3(pretrained=True, transform_input=False)
    
    target=9
    total_number_of_images = 64
    
    ACGen = ACGenerator(nz)
    ACGen.load_state_dict(torch.load('/content/ACGen_epoch_best.pth'))
    
    
    #select a batch of target
    label = np.asarray(trainset.targets)
    img=np.asarray(trainset.data/255)
    img_real=img[label==target][0:total_number_of_images]
    img_real=torch.tensor(img_real).permute(0,3,2,1)
    
    
    img_fake=acgan_calculate_IS(ACGen, target, nz, inception_model, total_number_of_images)
    vutils.save_image(img_fake.data, "./ac_images_fake_group_{}.png".format(target))
    ms_ssim_score = cal_ms_ssim(img_real, img_fake)
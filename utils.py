#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 23:40:32 2021

@author: shanyang
"""

from math import floor
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid

from pytorch_msssim import ms_ssim


def cal_ms_ssim(img_real,img_fake):
  up = nn.Upsample(size=(299,299), mode='bilinear',align_corners=False)
  img_real =up(img_real.type(torch.DoubleTensor))
  img_fake =up(img_fake.type(torch.DoubleTensor))

  ms_ssim_score = ms_ssim(img_fake,img_real,data_range=1,size_average=True)
  print('MS_SSIM avg {:.4f}'.format(ms_ssim_score))
  return ms_ssim_score



def calculate_inception_score(images,inception_model,n_split=10, eps=1E-16):
  inception_model.eval()
  up = nn.Upsample(size=(299,299), mode='bilinear',align_corners=False)
  images = up(images)
  yhat = F.softmax(inception_model(images))
  scores = list()
  n_part = floor(images.shape[0]/n_split)
  for i in range(n_split):
    ix_start, ix_end = i * n_part, i * n_part + n_part
    p_yx = yhat[ix_start:ix_end]
    p_y = p_yx.mean(dim=0).unsqueeze(0)
    kl_d = p_yx * (p_yx.log() - p_y.log())
    sum_kl_d = kl_d.sum(dim=1)
    avg_kl_d = sum_kl_d.mean()
    score = avg_kl_d.exp()
    scores.append(score.item())
  avg = np.mean(scores)
  std = np.std(scores)
  return avg, std



def cgan_calculate_IS(gen_model, target, nz, inception_model,total_number_of_images = 256):
  gen_model.to('cpu')
  
  class_target = torch.ones(total_number_of_images)*target

  with torch.no_grad():
      latent_z = torch.randn(total_number_of_images, nz)
      fake_img = gen_model(latent_z, class_target.int())

  avg, std = calculate_inception_score(fake_img,inception_model)
  print('IS avg {:.4f}, IS std {:.4f}'.format(avg, std))

  npimg = make_grid(fake_img, padding=0, nrow=16)
  fig, axes = plt.subplots(1,1, figsize=(10,10))
  axes.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
  fig.show()
    
  return fake_img



def acgan_calculate_IS(gen_model, target, nz, inception_model,total_number_of_images = 256):
  gen_model.to('cpu')
  class_target = torch.tensor([target])
  with torch.no_grad():
      latent_z = torch.randn(total_number_of_images, nz)
      latent_z[:,-10:]=F.one_hot(class_target, num_classes=10)
      fake_img = gen_model(latent_z)

  avg, std = calculate_inception_score(fake_img,inception_model)
  print('IS avg {:.4f}, IS std {:.4f}'.format(avg, std))

  npimg = make_grid(fake_img, padding=0, nrow=16)
  fig, axes = plt.subplots(1,1, figsize=(10,10))
  axes.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
  fig.show()
  
  return fake_img



def initialize_weights(m):
  if isinstance(m, nn.Conv2d):
    nn.init.normal_(m.weight.data, mean=0, std=0.02)
  elif isinstance(m, nn.BatchNorm2d):
    nn.init.normal_(m.weight.data, mean=1, std=0.02)
    nn.init.constant_(m.bias.data, 0)


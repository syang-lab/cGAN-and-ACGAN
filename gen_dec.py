#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 23:06:33 2021

@author: shanyang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class cGenerator(nn.Module):
  def __init__(self, nz):
    super(cGenerator, self).__init__()
    self.emb = nn.Embedding(10, 10)
    self.nz=nz
    
    self.model = nn.Sequential(
      nn.Linear(self.nz+10, 128),
			nn.BatchNorm1d(128, 0.8),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Linear(128, 256),
			nn.BatchNorm1d(256, 0.8),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Linear(256, 512),
			nn.BatchNorm1d(512, 0.8),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Linear(512, 1024),
			nn.BatchNorm1d(1024, 0.8),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Linear(1024, 3*32*32),
      nn.Tanh(),
			)
    
  def forward(self, nz, class_target):
   input = torch.cat((self.emb(class_target), nz), -1)
   img = self.model(input)
   img = img.view(img.size(0), 3, 32, 32)
   return img


class cDiscriminator(nn.Module):
  def __init__(self,nclass,img_size=32):
    super(cDiscriminator, self).__init__()
    self.emb = nn.Embedding(10, 10)
    self.nclass = nclass
    self.img_size = img_size
    
    self.model = nn.Sequential(
        nn.Linear(3082, 512),
        nn.Dropout(0.4),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Linear(512, 512),
        nn.Dropout(0.4),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Linear(512, 512),
        nn.Dropout(0.4),
        nn.LeakyReLU(0.2, inplace=True),
      
			  nn.Linear(512, 1),
			  nn.Sigmoid()
     )
    
  def forward(self, img, class_target):
    img = img.view(img.size(0),-1)
    input = torch.cat((img, self.emb(class_target)), -1)

    fc_isreal = self.model(input)
    return fc_isreal 



class ACDiscriminator(nn.Module):
    def __init__(self, nclass=10):
        super(ACDiscriminator, self).__init__()
      
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2,padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Dropout(p=0.5),

                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Dropout(p=0.5),

                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,stride=2,padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Dropout(p=0.5),

                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Dropout(p=0.5),

                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,stride=2,padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Dropout(p=0.5),

                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Dropout(p=0.5),
            )
            
        self.fc_linear = nn.Linear(8192,  256)
        self.fc_class =  nn.Linear(256, nclass)
        self.fc_isreal = nn.Linear(256, 1)


    def forward(self, input):
        bz=input.size()[0]
        out = self.conv(input)
        out=out.view(bz,-1)
        
        out = self.fc_linear(out)

        fc_class= F.softmax(self.fc_class(out))
        fc_isreal = F.sigmoid(self.fc_isreal(out))
        
        return fc_class, fc_isreal
    

class ACGenerator(nn.Module):
    def __init__(self, nz):
        super(ACGenerator, self).__init__()
        self.nz=nz
        
        #linear
        self.fc = nn.Linear(self.nz,384)
        
        self.tconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=384,out_channels=192, kernel_size=4, stride=1,padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=192,out_channels=96, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=96,out_channels=48, kernel_size=4, stride=2,padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=48,out_channels=3, kernel_size=4, stride=2,padding=1),
            nn.Tanh()
            )
        
        
    def forward(self, input):
        input = input.view(-1,self.nz)
        out = self.fc(input).view(-1,384,1,1)
        out = self.tconv(out)
       
        return out

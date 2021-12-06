#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 23:38:55 2021

@author: shanyang
"""
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.utils as vutils


def acgan_training(iterations, dataloader, dis_model, gen_model, optD, optG, class_loss, label_loss, nz, device):
    eval_z = torch.normal(0, 1, size=(32,nz))
    eval_class = torch.randint(0, 10, (32,))
    eval_z[:,-10:]=F.one_hot(eval_class, num_classes=10)
    eval_z =eval_z.to(device)

    for epoch in range(iterations):
        total_dis_loss=0
        total_gen_loss=0

        for img, class_target in dataloader:
            img = img.to(device)
            class_target = class_target.to(device)

            #train the discriminator
            optD.zero_grad()

            #real data is 1 and fake data is 0
            img_real_label = Variable(torch.ones(img.size()[0],1).to(device))
            img_fake_label = Variable(torch.zeros(img.size()[0],1).to(device))

            img_class_pred, img_label_pred = dis_model(img)
            loss_class_real= class_loss(img_class_pred, class_target)
            loss_label_real= label_loss(img_label_pred, img_real_label)
            loss_dis_real= loss_class_real+loss_label_real
            loss_dis_real.backward()
            optD.step()
            
            latent_z = torch.randn(img.size()[0],nz).to(device)
            class_target_fake =torch.randint(0,10, (img.size()[0],)).to(device)
            latent_z[:,-10:]=F.one_hot(class_target_fake, num_classes=10)

            fake_img = gen_model(latent_z)
            fake_img_class_pred, fake_img_label_pred=dis_model(fake_img.detach())
            
            loss_class_fake = class_loss(fake_img_class_pred, class_target_fake)
            loss_label_fake = label_loss(fake_img_label_pred, img_fake_label)
            loss_dis_fake=loss_class_fake +loss_label_fake
            loss_dis_fake.backward()
            optD.step()

            loss_dis = (loss_dis_fake+loss_dis_real)/2
 
            #train the generator
            gen_model.zero_grad()
            fake_img_class, fake_img_label = dis_model(fake_img)
            
            loss_class_gen = class_loss(fake_img_class, class_target)
            loss_label_gen = label_loss(fake_img_label, img_real_label)
            loss_gen =loss_class_gen +loss_label_gen
            
            loss_gen.backward()
            optG.step()
 
            total_dis_loss+=loss_dis.item()
            total_gen_loss+=loss_gen.item()

        print('Epoch {} Discriminator loss {:.2f} and Generator loss {:.2f}'.format(epoch, total_dis_loss/len(dataloader),total_gen_loss/len(dataloader)))
        with torch.no_grad():
          eval_img = gen_model(eval_z)
          vutils.save_image(eval_img.data, "eval_images_{}.png".format(epoch))
          vutils.save_image(img.data, "eval_real_images_{}.png".format(epoch))


        torch.save(gen_model.state_dict(), '/content/ACGen_epoch_best.pth')
        torch.save(dis_model.state_dict(), '/content/ACDis_epoch_best.pth')

    return 



def cgan_training(iterations, dataloader, dis_model, gen_model, optD, optG, label_loss, nz, device):
    eval_z = torch.normal(0, 1, size=(32,nz))
    eval_z =eval_z.to(device)
    eval_label = torch.randint(0, 10, (32,)).to(device)

    for epoch in range(iterations):
        total_dis_loss=0
        total_gen_loss=0

        for img, class_target in dataloader:
            img = img.to(device)
            class_target = class_target.to(device)

            #train the discriminator
            optD.zero_grad()

            #real data is 1 and fake data is 0
            img_real_label = Variable(torch.ones(img.size()[0],1).to(device))
            img_fake_label = Variable(torch.zeros(img.size()[0],1).to(device))

            img_label_pred = dis_model(img, class_target)

            loss_label_real= label_loss(img_label_pred, img_real_label)
            loss_label_real.backward()
            optD.step()
            
            latent_z = torch.randn(img.size()[0],nz).to(device)
            class_target_fake =torch.randint(0,10, (img.size()[0],)).to(device)

            fake_img = gen_model(latent_z, class_target_fake)
            fake_img_label_pred=dis_model(fake_img.detach(), class_target_fake)

            loss_label_fake = label_loss(fake_img_label_pred, img_fake_label)
            loss_label_fake.backward()
            optD.step()

            loss_dis = (loss_label_fake+loss_label_real)/2
 
 
            #train the generator
            gen_model.zero_grad()
            fake_img_label = dis_model(fake_img, class_target)
            loss_label_gen = label_loss(fake_img_label, img_real_label)
            loss_label_gen.backward()
            optG.step()

            total_dis_loss+=loss_dis.item()
            total_gen_loss+=loss_label_real.item()

        print('Epoch {} Discriminator loss {:.2f} and Generator loss {:.2f}'.format(epoch, total_dis_loss/len(dataloader),total_gen_loss/len(dataloader)))
        with torch.no_grad():
          eval_img = gen_model(eval_z, eval_label)
          vutils.save_image(eval_img.data, "c_eval_images_{}.png".format(epoch))

        torch.save(gen_model.state_dict(), '/content/cGen_epoch_best.pth')
        torch.save(dis_model.state_dict(), '/content/cDis_epoch_best.pth')

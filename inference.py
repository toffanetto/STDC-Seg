#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from models.model_stages import BiSeNet
from cityscapes import CityScapes
from unicampscapes import UnicampScapes
from cityscapes import get_class_colors

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import os.path as osp
import logging
import time
import numpy as np
from tqdm import tqdm
import math 

from PIL import Image
import matplotlib.pyplot as plt

def inference(dataset, respth='./checkpoints/train_STDC1-Seg/pths/model_maxmIOU50.pth', backbone='STDCNet813', scale=0.5, use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False, use_conv_last=False):
    print('scale', scale)
    print('use_boundary_2', use_boundary_2)
    print('use_boundary_4', use_boundary_4)
    print('use_boundary_8', use_boundary_8)
    print('use_boundary_16', use_boundary_16)
    ## dataset
    batchsize = 1
    n_workers = 2
    
    dl = DataLoader(dataset,
                    batch_size = batchsize,
                    shuffle = False,
                    num_workers = n_workers,
                    drop_last = False)

    n_classes = 19
    print("backbone:", backbone)
    net = BiSeNet(backbone=backbone, n_classes=n_classes,
     use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4, 
     use_boundary_8=use_boundary_8, use_boundary_16=use_boundary_16, 
     use_conv_last=use_conv_last)
    net.load_state_dict(torch.load(respth))
    net.cuda()
    net.eval()

    if dist.is_initialized() and dist.get_rank() != 0:
        diter = enumerate(dl)
    else:
        diter = enumerate(tqdm(dl))
        
    for i, (img, _, name,img_raw) in diter:

        imgs = img.cuda()

        N, C, H, W = imgs.size()
        
        size = (H, W)
        
        new_hw = [int(H*scale), int(W*scale)]

        imgs = F.interpolate(imgs, new_hw, mode='bilinear', align_corners=True)

        logits = net(imgs)[0]

        logits = F.interpolate(logits, size=size,
                mode='bilinear', align_corners=True) # Output vector
        probs = torch.softmax(logits, dim=1) # Probability for each class vector
        preds = torch.argmax(probs, dim=1) # Predicted class
        
        preds_tensor = preds[0].cpu()
        preds_np = preds_tensor.numpy() # Matrix of predicted classes 
        image_preds = get_class_colors(preds_np)
        im = Image.fromarray(image_preds, "RGB")
        im.save("./output/inference/prediction_"+str(name[0])+".png")
        
        img_raw = img_raw[0].numpy()
        
        plt.figure(figsize=(21,8))
        plt.subplot(1,3,1)
        plt.title('Input', fontsize = 16)
        plt.imshow(img_raw)
        plt.axis('off')
        plt.subplot(1,3,2)
        plt.imshow(image_preds)
        plt.title('Prediction', fontsize = 16)
        plt.axis('off')
        plt.tight_layout(pad=0.2)
        
        plt.savefig("./output/inference/sample_"+str(name[0])+".pdf",bbox_inches='tight')
        
        
if __name__ == "__main__":
    
    #Select dataset:
    #dataset = CityScapes('./data', mode='val')
    dataset = UnicampScapes('./data')
    
    inference(dataset=dataset, respth='./checkpoints/STDC1-Seg/model_maxmIOU75.pth', backbone='STDCNet813', 
              scale=0.75, use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)
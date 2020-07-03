# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 01:30:32 2019

@author: Onkar
"""
import cv2
import os
import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
from PIL import Image as im
classes = os.listdir("e:\\PDC\\CUB_200_2011\\CUB_200_2011\\images")
len(classes)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

tst_transform = transforms.Compose([
                                transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)])

model = models.densenet161(pretrained=False)
classifier = nn.Sequential(
  nn.Linear(in_features=2208, out_features=1024),
  nn.ReLU(),
  nn.Dropout(p=0.3),
  nn.Linear(in_features=1024, out_features=len(classes)),
  nn.LogSoftmax(dim=1)  
)
    
model.classifier = classifier
model.load_state_dict(torch.load('model.pt',map_location=torch.device('cpu')))

with torch.no_grad():
    model.eval()
    path = "5.jpg"
    imgss = cv2.imread(path)
    img = tst_transform(im.open(path))
    imgs = img[None, :, :, :]
    label_s = model.forward(imgs) 
    index = label_s.argmax(1).item()
    print(classes[index])

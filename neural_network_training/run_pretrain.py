###################################################################
# Created: 03/07/2023
# Last Modified: 03/07/2023 by Yujie Zhao
# Authors: Yujie Zhao, <your name>
# Emails: yzhao155@jh.edu
#
#
#### INPUTS:
# 1) list of slide image
# name,img,name
# Subset1_Train_30,Subset1_Train_30.tiff,Subset1_Train_30
# Subset1_Train_28,Subset1_Train_28.tiff,Subset1_Train_28
# Subset1_Train_75,Subset1_Train_75.tiff,Subset1_Train_75
# ......

#### OUTPUTS:
# 1) 1024 vector for each slide
# 2) label vector for each slide
########################################################
from Dataset.dataset2 import Dataset_name
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import os
from mtdp import build_model
from torchvision import transforms
import matplotlib.pyplot as plt
from numpy import asarray
from numpy import savetxt
import numpy as np
from torchmetrics import Accuracy
import openslide
import random
import pandas as pd
import random
from tempfile import TemporaryFile


######### transform
# imagenet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
# color norm. standard (from TCGA-A2-A3XS-DX1, Amgad et al, 2019): mean=[8.74108109, -0.12440419,  0.0444982], std=[0.6135447, 0.10989545, 0.0286032]
# val_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
#     ])

train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(scale=(.1, 0.5),degrees=5,translate=(0,0.5))

    ])
   

file = pd.read_csv("./file_list_subset1.txt")
file_list = file["img"]
print(file_list)

for j in file_list:
    print(j)
    img = Dataset_name(str("/home/yzhao155/data-acharl15/gleason_grading/patch/"+j.split('.tiff')[0]+"/patch_mask.jpg"), 
                        str("/home/yzhao155/data-acharl15/gleason_grading/patch/"+j.split('.tiff')[0]+"/patch_mask_"),
                        str('/home/yzhao155/data-acharl15/gleason_grading/'+j),transform = train_transform)
    print(img)
    train_loader = data_utils.DataLoader(img, batch_size=32, shuffle=True)

    print("load")
    # define a device which allows you to use GPU resources
    device = torch.device("cuda")
    model = build_model(arch="densenet121", pretrained="mtdp", pool=True)
    #model = torch.nn.DataParallel(model,device_ids = [0, 1, 2,3])
    model = model.to(device)
    print("model_inputed")

    with torch.no_grad():
        model.eval()
        # train
        features = list()
        classes = list()
        for i, (x, y) in enumerate(train_loader):
            #print("> train iter #{}".format(i + 1))
            out = model.forward(x.to(device))
            features.append(out.detach().cpu().numpy().squeeze())
            classes.append(np.asarray(y))

        features = np.vstack(features)
        classes = np.hstack(classes)


    outfile_feature = TemporaryFile()
    outfile_classes = TemporaryFile()
    print(features.shape,classes.shape)

    isExist = os.path.exists(str("/home/yzhao155/data-acharl15/gleason_grading/pretrain/"+j.split('.tiff')[0]))
    if not isExist:
        print("new_directory")
        #os.makedirs(str("/home/yzhao155/data-acharl15/gleason_grading/pretrain/"+j.split('.tiff')[0]))
    np.save(str("/home/yzhao155/data-acharl15/gleason_grading/pretrain/"+j.split('.tiff')[0]+"/augment_feature"), features)
    np.save(str("/home/yzhao155/data-acharl15/gleason_grading/pretrain/"+j.split('.tiff')[0]+"/augment_class"), classes)



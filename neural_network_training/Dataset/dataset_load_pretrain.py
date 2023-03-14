###################################################################
# Created: 03/07/2023
# Last Modified: 03/07/2023 by Yujie Zhao
# Authors: Yujie Zhao, <your name>
# Emails: yzhao155@jh.edu
#
#
#### INPUTS:
# 1) feature (1024 vector)
# 2) label

#### OUTPUTS:
# 1) put it into pytorch dataset

####
# dataset = Dataset_name(np.load("feature.npy"),np.load("class.npy"))
########################################################

import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np

class Dataset_name():
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label



    def __len__(self):
        'denotes the total number of samples'
        return len(self.label)

    def __getitem__(self, index):
        x = self.feature[index]
        y = self.label[index]
        return x, y
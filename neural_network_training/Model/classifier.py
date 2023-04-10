###################################################################
# Created: 03/07/2023
# Last Modified: 03/07/2023 by Yujie Zhao
# Authors: Yujie Zhao, <your name>
# Emails: yzhao155@jh.edu
#
# classifier defines the classifier structure.
#### INPUTS:
# 1) 1024 vector for each slide

#### OUTPUTS:
# 1) 5 vector
### Usage:
# model = Model_name()
########################################################
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class Model_name(nn.Module):
    # __init__ allows you to write the basic component of the model
    def __init__(self):
        super(Model_name,self).__init__()
        #super().__init__()
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1',nn.Linear(1024, 512)),
            ('relu',nn.ReLU()),
            ('dropout',nn.Dropout(p=0.5)),
            ('fc2',nn.Linear(512, 5))
            #('output',nn.Softmax(dim=1))     
        ]))
        # define your classifier
        # usually some linear layers with activation in between
    
    # forward tells how the data x go through the model
    def forward(self, x):
        y = self.classifier(x)
        return y

class tanh(nn.Module):
    # __init__ allows you to write the basic component of the model
    def __init__(self):
        super(tanh,self).__init__()
        #super().__init__()
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1',nn.Linear(1024, 512)),
            ('relu',nn.Tanh()),
            ('dropout',nn.Dropout(p=0.5)),
            ('fc2',nn.Linear(512, 5))
            #('output',nn.Softmax(dim=1))     
        ]))
        # define your classifier
        # usually some linear layers with activation in between
    
    # forward tells how the data x go through the model
    def forward(self, x):
        y = self.classifier(x)
        return y
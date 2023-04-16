# define your model here
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class Model_name(nn.Module):
    # __init__ allows you to write the basic component of the model
    def __init__(self, feature_encoder):
        super().__init__()
        self.feature_encoder = feature_encoder
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
        h = self.feature_encoder(x)
        #print(h.shape)
        y = self.classifier(h.view(h.shape[0],h.shape[1]*h.shape[2]*h.shape[3]))
        return y

from Model.model import Model_name
from Dataset.dataset import Dataset_name
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
import wandb
import random


# import wandb

# start a new wandb run to track this script


"""The openslide library helps you save memory but still load in image
# https://openslide.org/api/python/#module-openslide
# Example usage
# slide_ob = openslide.OpenSlide("/data/zwang/gleason-grading/Subset2_Train_1.tiff")
# print(slide_ob.dimensions)
# patch = np.array(slide_ob.read_region((0,0), 0, (256, 256)))[:,:,:3]  # (x,y), level, (width, height) 
# print(patch.shape)
"""

# choose which GPU you are going to use. we have 2
# notes: to check the GPU usage, type in "nvidia-smi" in terminal
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
 
# file_list = []
# for x in os.listdir("/data/zwang/gleason-grading/"):
#     if x.startswith("Subset2_Train"):
#         if x.endswith(".tiff"):
#         # Prints only text file present in My Folder
#             file_list.append(x)


file = pd.read_csv("./file_list_subset1.txt")
file_list = file["img"]
random.shuffle(file_list)

train_path_len = int(len(file_list)*0.8)

train_path = file_list[0:train_path_len]
val_path = file_list[train_path_len:len(file_list)]

#print(train_path,val_path)


#############
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])


#############
# define your dataset and data loader
# dataset define the data structure, training dataset and validation dataset usually have the some structure
train_dataset = []
for i in train_path:
    #print(i)
    tmp_path = Dataset_name(str("/home/yzhao155/data-acharl15/gleason_grading/patch/"+i.split('.tiff')[0]+"/patch_mask.jpg"), 
                      str("/home/yzhao155/data-acharl15/gleason_grading/patch/"+i.split('.tiff')[0]+"/patch_mask_"),
                     str('/home/yzhao155/data-acharl15/gleason_grading/'+i),transform = transform)
    #print(i,len(tmp_path))
    train_dataset = torch.utils.data.ConcatDataset([train_dataset,tmp_path])

val_dataset = []
for i in val_path:
    #print(i)
    tmp_path = Dataset_name(str("/home/yzhao155/data-acharl15/gleason_grading/patch/"+i.split('.tiff')[0]+"/patch_mask.jpg"), 
                      str("/home/yzhao155/data-acharl15/gleason_grading/patch/"+i.split('.tiff')[0]+"/patch_mask_"),
                     str('/home/yzhao155/data-acharl15/gleason_grading/'+i),transform = transform)
    #print(i,len(tmp_path))
    val_dataset = torch.utils.data.ConcatDataset([val_dataset,tmp_path])

print(len(val_dataset),len(train_dataset))
# dataloader define how you load data from the dataset. 
# There are important two aguments: (1) batch_size; (2) shuffle, it tells whether you want to shuffle the data. 
# We commonly set `shuffle = True`` in training and `shuffle = False` in validation

wandb.init(
    # set the wandb project where this run will be logged
    project="gleason-grading-subset1",name = 'preliminary-test3',
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 1e-4,
    "architecture": "CNN",
    "epochs": 100,
    }
)

train_loader = data_utils.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = data_utils.DataLoader(val_dataset, batch_size=32, shuffle=False)


# define a device which allows you to use GPU resources
device = torch.device("cuda")

# define your model
pretrained_model = build_model(arch="densenet121", pretrained="mtdp", pool= True)
model = Model_name(feature_encoder = pretrained_model)
#print(model)
### utilize GPU
model = torch.nn.DataParallel(model,device_ids = [0, 1, 2,3])
model = model.to(device)

param_update = []
for param in model.feature_encoder.parameters():
    param.requires_grad = False
    # tmp =random.randint(-1,18)
    # if tmp >= 0:
    #     number_false +=1
    #     param.requires_grad = False
    # else:
    #     param.requires_grad = True
    #     number_true += 1
#print("parameter",c)
for name,param in model.named_parameters():
    if param.requires_grad == True:
        print(name)
        param_update.append(param)

# ##########
# print("model_inputed")
# # define your optimizer and loss function
# criterion = nn.CrossEntropyLoss() # this is the loss function. Cross entropy loss is the most commonly used loss function for classification task
# optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay = 1e-4) # you could also use Adam optimizer if you want

# # define your validation function
# def validate(model, val_loader,classifier,accuracy):
#     model.eval()
#     val_loss = 0
#     accuracy_val = 0
#     # .....
#     with torch.no_grad():
#         for i, data in enumerate(val_loader,0):
#             inputs,labels = data
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             val_loss += criterion(outputs, labels)
#             output_label = classifier(outputs)
#             accuracy_val += accuracy(output_label.to(device),labels)
            
#     return val_loss/(i+1), accuracy_val/(i+1)

# # training process
# wandb.watch(model,log_freq=1)
# Loss_train = []
# Loss_val = []
# Accuracy_train = []
# Accuracy_val = []
# num_epochs = 100 # how many times you want to iterate the dataset

# for epoch in range(num_epochs):  # loop over the dataset multiple times
#     model.train()
#     print("running epoch",epoch)
#     loss_train = 0.0
#     accuracy_train = 0
#     for i, data in enumerate(train_loader, 0): # each time, load a batch of data from data loader
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data
#         # send your data and label to GPU device
#         inputs, labels = inputs.to(device), labels.to(device)
#         #print(labels)
#         # zero the parameter gradients
#         optimizer.zero_grad()
#         # forward + backward + optimize
#         outputs= model(inputs)
#         classifier = nn.Softmax(dim=1)
#         output_label = classifier(outputs)
#         #print(output_label)
#         #print(output_label)
#         loss = criterion(outputs, labels)
#         # print("batch",i, loss)
#         loss.backward()
#         optimizer.step()
#         accuracy = Accuracy(task="multiclass", num_classes=5).to(device)
#         accuracy_train += accuracy(output_label.to(device),labels)
#         loss_train += loss.item()

        
#     # once you are done with iterating the dataset once, 
#     # check loss, 
#     loss_train = loss_train/(i+1) # calculate the avergaed loss over all batches
#     loss_val, accuracy_val = validate(model, val_loader,classifier,accuracy)
#     accuracy_train = accuracy_train/(i+1)
#     # let's print these loss out
#     print("epoch {}, training loss = {:.3f}, validation loss = {:.3f}, training accuracy = {:.3f}, validation accuracy = {:.3f}".format(
#         epoch,
#         loss_train,
#         loss_val,
#         accuracy_train,
#         accuracy_val 
#     ))
#     # and log loss
#     wandb.log({"accuracy_train": accuracy_val, "loss_train": loss_train,"accuracy_val": accuracy_val, "loss_val": loss_val})
#     Loss_train.append(loss_train)
#     Loss_val.append(loss_val)
#     Accuracy_train.append(accuracy_train)
#     Accuracy_val.append(accuracy_val)
#     state_dict = model.state_dict()
#     torch.save(state_dict, str('./result/subset1_100epoch/checkpoint/checkpoint_epoch{}.pth'.format(epoch)))

# wandb.finish()
# print('Finished Training')


# np.savetxt('./result/subset1_100epoch/Loss_train.txt', Loss_train, delimiter='\t',fmt='%s')
# np.savetxt('./result/subset1_100epoch/Loss_val.txt', Loss_val, delimiter='\t',fmt='%s')
# np.savetxt('./result/subset1_100epoch/Accuracy_train.txt', Accuracy_train, delimiter='\t',fmt='%s')
# np.savetxt('./result/subset1_100epoch/Accuracy_val.txt', Accuracy_val, delimiter='\t',fmt='%s')
# # plot loss
# plt.figure()
# plt.plot(list(range(num_epochs)),Loss_train,label="train")
# plt.plot(list(range(num_epochs)),Loss_val,label="validation")
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.legend()
# plt.savefig("./result/subset1_100epoch/loss.pdf")

# # plot accuracy
# plt.figure()
# plt.plot(list(range(num_epochs)),Accuracy_train,label="train")
# plt.plot(list(range(num_epochs)),Accuracy_val,label="validation")
# plt.xlabel("epoch")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.savefig("./result/subset1_100epoch/Accuracy.pdf")
###################################################################
# Created: 03/07/2023
# Last Modified: 03/07/2023 by Yujie Zhao
# Authors: Yujie Zhao, <your name>
# Emails: yzhao155@jh.edu
#
# classifier defines the classifier structure.
#### INPUTS:
# 1) list of slide image
# name,img,name
# Subset1_Train_30,Subset1_Train_30.tiff,Subset1_Train_30
# Subset1_Train_28,Subset1_Train_28.tiff,Subset1_Train_28
# Subset1_Train_75,Subset1_Train_75.tiff,Subset1_Train_75
# ......
# 2) output directory

#### OUTPUTS
# 1) parameters for each epoch
# 2) loss, accuracy for train & test for each epoch

#### Run the script: nohup python train_classifier.py &
########################################################
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
from Model.classifier import Model_name,tanh
from Dataset.dataset_load_pretrain import Dataset_name
import torch.utils.data as data_utils
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
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
import pickle
import wandb


#################### output
out_dir = "/data/acharl15/gleason_grading/result/training_saved_model/subset3_pretrain_100epoch_normalized_Imagenet_balanced_relu_dropout_SGD_augment/"
isExist = os.path.exists("/data/acharl15/gleason_grading/result/training_saved_model/subset3_pretrain_100epoch_normalized_Imagenet_balanced_relu_dropout_SGD_augment/checkpoint")
if not isExist:
    print("new_directory")
    os.makedirs("/data/acharl15/gleason_grading/result/training_saved_model/subset3_pretrain_100epoch_normalized_Imagenet_balanced_relu_dropout_SGD_augment/checkpoint")

#################### input 
file = pd.read_csv("/data/acharl15/gleason_grading/file_list_all.csv")
file = file.dropna()
file_list = []
for i in file["img"]:
    if i.startswith("Subset3"):
        file_list.append(i)
    #file_list.append(i)
file_list
random.seed(42)
random.shuffle(file_list)

train_path_len = int(len(file_list)*0.8)
train_path = file_list[0:train_path_len]
val_path = file_list[train_path_len:len(file_list)]

train_val = []
train_val.append(train_path)
train_val.append(val_path)
# save train val split
print('\nSaving the object')
with open(str(out_dir+"train_val_split.pck"), "wb") as output_file:
    pickle.dump(train_val, output_file)

print(train_path,val_path)

# define your dataset and data loader
# dataset define the data structure, training dataset and validation dataset usually have the some structure
train_dataset = []
label_list = []
n = 0
for i in train_path:
    aug1_feature = np.load(str("/home/yzhao155/data-acharl15/gleason_grading/pretrain/"+i+"/augment2_feature.npy"))
    aug1_label = np.load(str("/home/yzhao155/data-acharl15/gleason_grading/pretrain/"+i+"/augment2_class.npy"))
    aug2_feature = np.load(str("/home/yzhao155/data-acharl15/gleason_grading/pretrain/"+i+"/augment3_feature.npy"))
    aug2_label = np.load(str("/home/yzhao155/data-acharl15/gleason_grading/pretrain/"+i+"/augment3_class.npy"))
    origin_feature = np.load(str("/home/yzhao155/data-acharl15/gleason_grading/pretrain/"+i+"/feature.npy"))
    origin_label = np.load(str("/home/yzhao155/data-acharl15/gleason_grading/pretrain/"+i+"/class.npy"))
    feature = np.vstack((aug1_feature,aug2_feature,origin_feature))
    label = np.hstack((aug1_label,aug2_label,origin_label))
    #print(feature.shape,label.shape)
    tmp_path = Dataset_name(feature,label)
    for j in label:
        label_list.append(j)
    #print(i,len(tmp_path))
    n+=len(label)
    train_dataset = torch.utils.data.ConcatDataset([train_dataset,tmp_path])

val_dataset = []
for i in val_path:
    feature = np.load(str("/home/yzhao155/data-acharl15/gleason_grading/pretrain/"+i+"/feature.npy"))
    label = np.load(str("/home/yzhao155/data-acharl15/gleason_grading/pretrain/"+i+"/class.npy"))
    #print(feature.shape,label.shape)
    tmp_path = Dataset_name(feature,label)
    #print(i,len(tmp_path))
    val_dataset = torch.utils.data.ConcatDataset([val_dataset,tmp_path])


print(len(val_dataset),len(train_dataset))
train_loader = data_utils.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = data_utils.DataLoader(val_dataset, batch_size=32, shuffle=True)

# dataloader define how you load data from the dataset. 
# There are important two aguments: (1) batch_size; (2) shuffle, it tells whether you want to shuffle the data. 
# We commonly set `shuffle = True`` in training and `shuffle = False` in validation
##################### set wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="gleason-grading-subset1",name = 'Experiment 10_2',
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 1e-4,
    "architecture": "CNN",
    "epochs": 100,
    }
)


####################################define model
# define a device which allows you to use GPU resources
device = torch.device("cuda")
# define your model
model = Model_name()
### utilize GPU
param=torch.load("/data/acharl15/gleason_grading/result/training_saved_model/subset3_pretrain_100epoch_normalized_Imagenet_balanced_relu_dropout_SGD_augment/checkpoint/checkpoint_epoch61.pth")
#print(param)
model_dict = model.state_dict()
pretrained_dict = {k.partition('module.')[2]: v for k, v in param.items() if k.partition('module.')[2] in model_dict}
#print(pretrained_dict)
model_dict.update(pretrained_dict) 
# 3. load the new state dict
model.load_state_dict(model_dict)
print("loaded model")
### utilize GPU 
#print(model)
model = torch.nn.DataParallel(model)
model = model.to(device)
print("model_inputed")
################################ define your optimizer and loss function
label_list = np.array(label_list)
class1 = 1-len(label_list[label_list== 0])/n
class2 = 1-len(label_list[label_list== 1])/n
class3 = 1-len(label_list[label_list== 2])/n
class4 = 1-len(label_list[label_list== 3])/n
class5 = 1-len(label_list[label_list== 4])/n
class_weight = torch.FloatTensor([class1,class2,class3,class4,class5]).to(device)
print("class weight",class_weight)
criterion = nn.CrossEntropyLoss(weight = class_weight) # this is the loss function. Cross entropy loss is the most commonly used loss function for classification task
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay = 1e-4) # you could also use Adam optimizer if you want
# output parameter
# f = open(str(out_dir+"parameter.txt"), "a")
# f.write(str("criterion:"+"CrossEntropyLoss"+str(class_weight)+"\n"))
# f.write(str("optimizer:"+"SGD"+"lr=1e-4, momentum=0.9, weight_decay = 1e-4 \n"))
# f.write("epoch: 100 \n")
# f.write(str("train: "+str(len(train_dataset))+" val: "+str(len(val_dataset))+"\n"))
# f.close()
################################# define your validation function
def validate(model, val_loader,classifier,accuracy):
    model.eval()
    val_loss = 0
    accuracy_val = 0
    # .....
    with torch.no_grad():
        for i, data in enumerate(val_loader,0):
            inputs,labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels)
            output_label = classifier(outputs)
            accuracy_val += accuracy(output_label.to(device),labels)
            
    return val_loss/(i+1), accuracy_val/(i+1)

###################################### training process
wandb.watch(model,log_freq=1)
Loss_train = []
Loss_val = []
Accuracy_train = []
Accuracy_val = []
num_epochs = 100 # how many times you want to iterate the dataset

for epoch in range(62,500):  # loop over the dataset multiple times
    model.train()
    print("running epoch",epoch)
    loss_train = 0.0
    accuracy_train = 0
    #train_output_label = list()
    for i, data in enumerate(train_loader, 0): # each time, load a batch of data from data loader
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # send your data and label to GPU device
        inputs, labels = inputs.to(device), labels.to(device)
        #print(labels)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs= model(inputs)
        classifier = nn.Softmax(dim=1)
        output_label = classifier(outputs)
        #print(output_label)
        #print(output_label)
        loss = criterion(outputs, labels)
        # print("batch",i, loss)
        loss.backward()
        optimizer.step()
        accuracy = Accuracy(task="multiclass", num_classes=5).to(device)
        accuracy_train += accuracy(output_label.to(device),labels)
        #train_output_label.append(output_label.detach().cpu().numpy().squeeze())
        loss_train += loss.item()

        
    # once you are done with iterating the dataset once, 
    # check loss, 
    loss_train = loss_train/(i+1) # calculate the avergaed loss over all batches
    loss_val, accuracy_val = validate(model, val_loader,classifier,accuracy)
    accuracy_train = accuracy_train/(i+1)
    # let's print these loss out
    print("epoch {}, training loss = {:.3f}, validation loss = {:.3f}, training accuracy = {:.3f}, validation accuracy = {:.3f}".format(
        epoch,
        loss_train,
        loss_val,
        accuracy_train,
        accuracy_val 
    ))
    # and log loss
    wandb.log({"accuracy_train": accuracy_train, "loss_train": loss_train,"accuracy_val": accuracy_val, "loss_val": loss_val})
    Loss_train.append(loss_train)
    Loss_val.append(loss_val)
    Accuracy_train.append(accuracy_train)
    Accuracy_val.append(accuracy_val)
    state_dict = model.state_dict()
    torch.save(state_dict, str(out_dir+'checkpoint/checkpoint_epoch{}.pth'.format(epoch)))

wandb.finish()
print('Finished Training')

# ############################ output
# Loss_train = np.array(Loss_train)
# Loss_val = np.array(Loss_val)s
# Accuracy_train = np.array(Accuracy_train)
# Accuracy_val = np.array(Accuracy_val)
# np.savetxt(str(out_dir+'Loss_train.txt'), Loss_train, delimiter='\t',fmt='%s')
# np.savetxt(str(out_dir+'Loss_val.txt'), Loss_val, delimiter='\t',fmt='%s')
# np.savetxt(str(out_dir+'Accuracy_train.txt'), Accuracy_train, delimiter='\t',fmt='%s')
# np.savetxt(str(out_dir+'Accuracy_val.txt'), Accuracy_val, delimiter='\t',fmt='%s')
# # plot loss
# plt.figure()
# plt.plot(list(range(num_epochs)),Loss_train,label="train")
# plt.plot(list(range(num_epochs)),Loss_val,label="validation")
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.legend()
# plt.savefig(str(out_dir+'loss.pdf'))

# # plot accuracy
# plt.figure()
# plt.plot(list(range(num_epochs)),Accuracy_train,label="train")
# plt.plot(list(range(num_epochs)),Accuracy_val,label="validation")
# plt.xlabel("epoch")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.savefig(str(out_dir+"Accuracy.pdf"))
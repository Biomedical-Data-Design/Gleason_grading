import torch
import torchvision
import torchvision.transforms as transforms
import os
import pandas as pd
from torchvision.io import read_image
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']= pow(2,40).__str__()
import cv2
import numpy as np
import openslide

class Dataset_name():
    def __init__(self, patch_mask_path, patch_annotation_path,img_path, transform=None):
        self.transform = transform
        #img = cv2.imread(img_path,1)
        # input must be float 32
        self.slide_ob = openslide.OpenSlide(img_path)
        patch_mask = cv2.imread(patch_mask_path,0)
        ret,self.thresh = cv2.threshold(patch_mask,127,255,cv2.THRESH_BINARY)
        #print(thresh.shape)
        self.patch_annotation = {"Normal":0,
                         "Stroma":0,
                         "G3":0,
                         "G4":0,
                         "G5":0}
        annotation_sum = np.zeros(self.thresh.shape,dtype=int)
        for a in range(5):
            p = cv2.imread(str(patch_annotation_path+list(self.patch_annotation.keys())[a]+".jpg"),0)
            ret,ann = cv2.threshold(p,127,255,cv2.THRESH_BINARY)
            annotation_sum = np.add(annotation_sum,ann)
            self.patch_annotation[list(self.patch_annotation.keys())[a]]=ann
            
        self.ROW, self.COL = np.where((self.thresh>0)&(annotation_sum == 255))
        #print(len(self.ROW),len(self.COL))

        # #randomly select 20%
        # total_patch = np.array(range(len(self.ROW)))
        # np.random.shuffle(total_patch)
        # number = int(len(self.ROW)*0.2)
        # total_patch = total_patch[0:number]s
        # #print(self.ROW[total_patch[0:2]])
        # self.ROW = self.ROW[total_patch.tolist()]
        # self.COL = self.COL[total_patch.tolist()]
        # #print(self.ROW[0:2])



    def __len__(self):
        'denotes the total number of samples'
        return len(self.ROW)

    def __getitem__(self, index):
        row, col = self.ROW[index], self.COL[index]
        label_vector = [self.patch_annotation['Normal'][row, col],
             self.patch_annotation['Stroma'][row, col],
             self.patch_annotation['G3'][row, col],
             self.patch_annotation['G4'][row, col],
             self.patch_annotation['G5'][row, col]]
        
        # relocate on the original img
        row_left_top = int((self.slide_ob.dimensions[1]-self.thresh.shape[0]*256)/2)
        col_left_top = int((self.slide_ob.dimensions[0]-self.thresh.shape[1]*256)/2)
        row_start = int(row_left_top+row*256)
        col_start = int(col_left_top+col*256)
        x = np.array(self.slide_ob.read_region((col_start,row_start),0, (256, 256)))[:,:,:3] 
        
        y = label_vector.index(255)
        if self.transform is not None:
            x = self.transform(x)
        idx = [row,col]
        return x, idx, y

class Dataset_name_test():
    def __init__(self, patch_mask_path, patch_annotation_path,img_path, transform=None):
        self.transform = transform
        #img = cv2.imread(img_path,1)
        # input must be float 32
        self.slide_ob = openslide.OpenSlide(img_path)
        patch_mask = cv2.imread(patch_mask_path,0)
        ret,self.thresh = cv2.threshold(patch_mask,127,255,cv2.THRESH_BINARY)
        #print(thresh.shape)
        self.patch_annotation = {"Normal":0,
                         "Stroma":0,
                         "G3":0,
                         "G4":0,
                         "G5":0}
        annotation_sum = np.zeros(self.thresh.shape,dtype=int)
        for a in range(5):
            p = cv2.imread(str(patch_annotation_path+list(self.patch_annotation.keys())[a]+".jpg"),0)
            ret,ann = cv2.threshold(p,127,255,cv2.THRESH_BINARY)
            annotation_sum = np.add(annotation_sum,ann)
            self.patch_annotation[list(self.patch_annotation.keys())[a]]=ann
            
        self.ROW, self.COL = np.where(self.thresh>0)

    def __len__(self):
        'denotes the total number of samples'
        return len(self.ROW)

    def __getitem__(self, index):
        row, col = self.ROW[index], self.COL[index]
        label_vector = [self.patch_annotation['Normal'][row, col],
             self.patch_annotation['Stroma'][row, col],
             self.patch_annotation['G3'][row, col],
             self.patch_annotation['G4'][row, col],
             self.patch_annotation['G5'][row, col]]
        
        # relocate on the original img
        row_left_top = int((self.slide_ob.dimensions[1]-self.thresh.shape[0]*256)/2)
        col_left_top = int((self.slide_ob.dimensions[0]-self.thresh.shape[1]*256)/2)
        row_start = int(row_left_top+row*256)
        col_start = int(col_left_top+col*256)
        x = np.array(self.slide_ob.read_region((col_start,row_start),0, (256, 256)))[:,:,:3] 
        
        #y = label_vector.index(255)
        if self.transform is not None:
            x = self.transform(x)
        idx = [row,col]
        return x, idx, label_vector
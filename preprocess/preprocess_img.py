##path
import os
import re
import sys
from PIL import Image 
#from skimage.util.shape import view_as_windows
#from skimage import io
import numpy as np
from patchify import patchify 
#import imageio

Image.MAX_IMAGE_PIXELS = 100000000000 
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']= pow(2,40).__str__()
import cv2
##### get path
import argparse
from argparse import ArgumentParser
import openslide

parser = ArgumentParser(description="Preprocessed File")
parser.print_help()
parser.add_argument("--i", dest="filename", required=False,
    help="input original image", metavar="FILE", default = 'Subset1_Train_1.tiff')
parser.add_argument('--annotation', dest="annotation_dir", required=False,
    help="annotation directory", default = 'Subset1_Train_1')
parser.add_argument('--output', dest="output_dir", required=False,
    help="output directory",default = 'Subset1_Train_1')
args = parser.parse_args()


print(args.filename,args.annotation_dir,args.output_dir)


image = os.path.join("/home/yzhao155/data-acharl15/gleason_grading_yujie/"+ args.filename)
ann_dir = os.path.join('/home/yzhao155/data-acharl15/gleason_grading/Subset1_Train_annotation/'+ args.annotation_dir+"/")
output = os.path.join("/home/yzhao155/data-acharl15/gleason_grading_yujie/patch/"+args.output_dir+"/")

print(image, ann_dir, output)

annotation_image = []
for i in os.listdir(ann_dir):
    if i.endswith('tif'):
        annotation_image.append(os.path.join(ann_dir,i))
print(annotation_image)

# transfer BGR format to gray format
if os.path.exists(image):
    print("path exists")
else:
    print("no pathway")

print(type(image))
img = cv2.imread(str("/home/yzhao155/data-acharl15/gleason_grading_yujie/"+ args.filename),1)
#img=openslide.OpenSlide(str("/home/yzhao155/data-acharl15/gleason_grading_yujie/"+ args.filename))
#print(img.dimensions)
#print(type(img))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.threshold(): choose a threshold (thresh_ostu choose a value between two peaks), https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
#thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]: make graph white&black
#https://blog.csdn.net/lcalqf/article/details/71170171
otsu,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#how many patch can be make
row_num = int(thresh.shape[0]/256)
col_num = int(thresh.shape[1]/256)

print(row_num, col_num)
# cut edge
row_left_top = int((thresh.shape[0]-row_num*256)/2)
row_left_bottom = int((thresh.shape[0]-row_num*256)-row_left_top)
col_left_top = int((thresh.shape[1]-col_num*256)/2)
col_left_bottom = int((thresh.shape[1]-col_num*256)-col_left_top)

patch_mask = np.zeros([row_num,col_num],dtype=int)
patch_mask_annotation = np.zeros([5,row_num,col_num],dtype=int)

patches_img = patchify(thresh[int(row_left_top):int(thresh.shape[0]-row_left_bottom),
                           int(col_left_top):int(thresh.shape[1]-col_left_bottom)], (256, 256), step=256)

annotation_mask = []
order_annotation = []
for i in annotation_image:
    img_annotate = cv2.imread(i,0)
    gleason_class = os.path.basename(i).replace('_Mask.tif','')
    order_annotation.append(gleason_class)
    annotation_mask.append(img_annotate)


def align_annotate(sinlgle_patch_index,annotation_mask, order_annotation):
    annotate_dict = {'Normal':0,'Stroma': 0,'G3': 0,'G4': 0,'G5': 0}
    for i in range(len(annotation_mask)):
        img_annotate = annotation_mask[i]
        gleason_class = order_annotation[i]
        patch_annotation = img_annotate[int(sinlgle_patch_index[0]):int(sinlgle_patch_index[1]),
                                        int(sinlgle_patch_index[2]):int(sinlgle_patch_index[3])]
        # how many pixel showed in each class
        if len(np.nonzero(patch_annotation)[0]) > 0 :
            annotate_dict[gleason_class] = 1
    return annotate_dict

count=0
for i in range(patches_img.shape[0]):
    for j in range(patches_img.shape[1]):
        single_patch_img = patches_img[i, j,:,:]
        # on thresh, count how many pixel are white
        non_zero = len(np.nonzero(single_patch_img)[0])
        # if the prevalence are less than 0.01%, delete patches
        if (non_zero / (256*256)) > 0.01:
            row_start = int(row_left_top+i*256)
            col_start = int(col_left_top+j*256)
            # index on the original 
            single_patch=[row_start, int(row_start+256),col_start,int(col_start+256)]
            annotation = align_annotate(single_patch, annotation_mask,order_annotation)
            if np.sum(list(annotation.values())) > 0:
                patch_mask[i,j]=1
                for k in range(len(annotation.keys())):
                    #print("have annotation")
                    patch_mask_annotation[k][i,j]=list(annotation.values())[k]


patch_mask_output = cv2.convertScaleAbs(patch_mask, alpha=(255.0))
cv2.imwrite(str(output+"patch_mask.jpg"), patch_mask_output)

for i in range(5):
    patch_mask_annotation[i]= cv2.convertScaleAbs(patch_mask_annotation[i], alpha=(255.0))

cv2.imwrite(str("/home/yzhao155/data-acharl15/gleason_grading_yujie/patch/"+args.output_dir+"/patch_mask_Normal.jpg"), patch_mask_annotation[0])
cv2.imwrite(str("/home/yzhao155/data-acharl15/gleason_grading_yujie/patch/"+args.output_dir+"/patch_mask_Stroma.jpg"), patch_mask_annotation[1])
cv2.imwrite(str("/home/yzhao155/data-acharl15/gleason_grading_yujie/patch/"+args.output_dir+"/patch_mask_G3.jpg"), patch_mask_annotation[2])
cv2.imwrite(str("/home/yzhao155/data-acharl15/gleason_grading_yujie/patch/"+args.output_dir+"/patch_mask_G4.jpg"), patch_mask_annotation[3])
cv2.imwrite(str("/home/yzhao155/data-acharl15/gleason_grading_yujie/patch/"+args.output_dir+"/patch_mask_G5.jpg"), patch_mask_annotation[4])

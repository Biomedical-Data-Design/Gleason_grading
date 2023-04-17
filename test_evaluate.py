
from preprocess.preprocess_img_function import preprocess_test
from evaluate import Evaluate
import os
##### get path
import argparse
from argparse import ArgumentParser

parser = ArgumentParser(description="Preprocess + Run model")
parser.print_help()

parser.add_argument("--i", dest="img_path", required=False,
    help="input original image", metavar="FILE", default = '/data/acharl15/gleason_grading/test_folder/Subset1_Test_40.tiff')
parser.add_argument('--annotation', dest="ann_path", required=False,
    help="annotation directory", default = '/data/acharl15/gleason_grading/test_folder/Subset1_Test_annotation/Subset1_Test_40/')
parser.add_argument('--output_patch_folder', dest="output_patch_folder", required=False,
    help="output patch directory",default = '/data/acharl15/gleason_grading/test_folder/patch/Subset1_Test_40/')
parser.add_argument('--output_result_folder', dest="out_dir", required=False,
    help="output result directory",default = '/data/acharl15/gleason_grading/test_folder/result/Subset1_Test_40/')
parser.add_argument('--model', dest="model", required=False,
    help="model",default = "./checkpoint/Subset1_epoch36.pth")
args = parser.parse_args()

print(args.img_path,args.ann_path,args.output_patch_folder)

################### preprocess #################
img_path = args.img_path
ann_path = args.ann_path
output_patch_folder = args.output_patch_folder
out_dir = args.out_dir

model = args.model
preprocess_test(img_path,ann_path,output_patch_folder)
Evaluate(model,out_dir,output_patch_folder,img_path)
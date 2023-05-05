
import sys 
sys.path.append("../")
from preprocess.preprocess_img_function import preprocess_image
from test_evaluation.run_model import Evaluate
import argparse
from argparse import ArgumentParser

parser = ArgumentParser(description="Automate Gleason Grading System")
parser.print_help()

parser.add_argument("--i", dest="img_path", required=False,
    help="input original image", metavar="FILE", default = '/data/acharl15/gleason_grading/test_folder/Subset1_Test_24.tiff')
parser.add_argument('--output_folder', dest="out_dir", required=False,
    help="output result directory",default = './demo_result/')
parser.add_argument('--model', dest="model", required=False,
    help="model",default = "../neural_network_training/checkpoint/Subset1_epoch36.pth")
parser.add_argument('--background', dest="bg", required=False,
    help="background color",default = "white")
args = parser.parse_args()
################### preprocess #################
#img_path = "/data/acharl15/gleason_grading/test_folder/Subset1_Test_24.tiff"
#output_folder = "./demo_result/"
# if your scan are in black, change bg = "black"
#preprocess_image(args.img_path,args.out_dir,bg=args.bg)
#################run model ######################
#model = "./checkpoint/Subset1_epoch36.pth"
Evaluate(args.model,args.out_dir,args.out_dir,args.img_path)

from preprocess.preprocess_img_function import preprocess
from evaluate import Evaluate

################### preprocess #################
img_path = "/data/acharl15/gleason_grading/test_folder/Subset1_Test_24.tiff"
ann_path = "/data/acharl15/gleason_grading/test_folder/Subset1_Test_annotation/Subset1_Test_24/"
output_patch_folder = "./demo_result/patch/"
preprocess(img_path,ann_path,output_patch_folder)


#################run model ######################
model = "/data/acharl15/gleason_grading/result/training_saved_model/subset1_pretrain_100epoch_normalized_Imagenet_balanced_relu_dropout_SGD_augment/checkpoint/checkpoint_epoch36.pth"
out_dir = "./demo_result/result/"
Evaluate(model,out_dir,output_patch_folder,img_path)
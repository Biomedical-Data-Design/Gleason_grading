# Gleason_grading

## This repository contains the relevant code to utilize a Convolutional Neural Network to classify the Gleason grades of prostate cancer. 

The image database is sourced from the Automated Gleason Grading Challenge 2022 (AGGC2022). 

## To run the model, do the following:
1. copy the github repo:   `git clone https://github.com/Biomedical-Data-Design-2022-2023/Gleason_grading.git`
2. create an environment: `conda env create -f environment.yml`
3. Go to demo folder
`python demo.py --i /data/acharl15/gleason_grading/test_folder/Subset1_Test_24.tiff --output_folder ./demo_result/ --model ../neural_network_training/checkpoint/Subset1_epoch36.pth --background "white"`
--i: define original image path
--output_folder: define result path
--model: define pretrain model path (saved under ./checkpoint/)
--background: define what is the color of the background of the same scans ("white","black"). 

The output contains:
1. patch_mask.jpg: binary image indicating tissue region
2. G_pred.jpg: Scalar Image, Absolute classification of each patch image. Whiter color means higher grading score. Size of height x weight (where the pixel value is the predicted class label) (0:empty background, 3. 1:normal, etc)
3. G_pred_color.jpg: Colored Heatmap based on G_pred.jpg
4. G_pred_after_mor.jpg: Scalar Image after closing morphological transformation. Size of height x weight (where the pixel value is the predicted class label) (0:empty background, 3. 1:normal, etc)
5. G_pred_after_mor_color.jpg:  Colored Heatmap based on G_pred_after_mor.jpg
6. result.pck  : saved y-probability, true label, and index in pickel file, which is a dictionary.
    output[“yprob”] = yprob (size: n*5)
    output[“ytrue”] = label (size: n)
    output[“index_x”] = index_list_x (size: n)
    output[“index_y”] = index_list_y (size: n)
Colored image label: Green: normal; Blue: stroma; Yellow: G3; Fuchsia: G4; Red :G5
Each pixel value is corrected by confidence level, where ambiguous color assignment indicates lower confidence level of class assignment.


The automated grading system contains the following 2 steps:
### 1. Preprocessing Step: See folder preprocess
### 2.  Neural Network Training: see folder neural_network_training
* Trained model parameters are saved under the `neural_network_training/checkpoint` folder


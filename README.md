# Gleason_grading

This repository contains the relevant code to utilize a Convolutional Neural Network to classify the Gleason grades of prostate cancer. 

The image database is sourced from the Automated Gleason Grading Challenge 2022 (AGGC2022). 

1. Preprocessing Step: 

Inputs: Original slide image, annotation masks
Outputs: Patch masks for the complete image and all 5 classes 

preprocess_img.py contains all preprocessing steps including 
	- Image segmentation into 256 x 256 patches using the Patchify library
    - Image normalization (???)
    
Additionally, for images that OpenCV is not suited to load in, there are two other versions: preprocess_img_openslide.py uses the Openslide library to load in images, and preprocess_img_pyvips.py uses the Pyvips library.
 
The directory names and appropriate library file locations must be updated within the code.

### 1. Preprocessing Step: See folder preprocess
Inputs: Original slide image, annotation masks, output_patch_folder  
Outputs: Patch masks for the complete image and all 5 classes 

`preprocess_img.py` contains all preprocessing steps including 
* Image segmentation into 256 x 256 patches using the Patchify library
    
Additionally, for images that OpenCV is not suited to load in, there are two other versions: `preprocess_img_openslide.py` uses the Openslide library to load in images, and `preprocess_img_pyvips.py` uses the Pyvips library.
 
The directory names and appropriate library file locations must be updated within the `demo.py`.

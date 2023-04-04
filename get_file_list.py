import pandas as pd
import numpy as np
import os

img = []
img_path = []
img_ann = []
img_pretrain = []
ann_file = []
pretrain_file = []
for i in os.listdir("/data/acharl15/gleason_grading/"):
    if i.endswith(".tiff"):
        img.append(i.replace('.tiff', ''))
        img_path.append(str("/data/acharl15/gleason_grading/"+i))
        tmp_patch = str("/data/acharl15/gleason_grading/patch/"+i.replace('.tiff', '')+"/")
        if os.path.exists(tmp_patch):
            img_ann.append(tmp_patch)
            tmp_patch_file = os.listdir(str("/data/acharl15/gleason_grading/patch/"+i.replace('.tiff', '')+"/"))
            ann_file.append(tmp_patch_file)
        else:
            img_ann.append(np.nan)
            ann_file.append(np.nan)
        
        tmp_pretrain = str("/data/acharl15/gleason_grading/pretrain/"+i.replace('.tiff', '')+"/")
        if os.path.exists(tmp_pretrain):
            img_pretrain.append(tmp_pretrain)
            tmp_pretrain= os.listdir(str("/data/acharl15/gleason_grading/pretrain/"+i.replace('.tiff', '')+"/"))
            pretrain_file.append(tmp_pretrain)
        else:
            img_pretrain.append(np.nan)
            pretrain_file.append(np.nan)



file = pd.DataFrame()
file["img"] = img
file["img_path"] = img_path
file["img_annotation"] = img_ann
file["img_pretrain"] = img_pretrain
file["ann_file"] = ann_file
file["pretrain_file"] = pretrain_file


print(file[0:10])
file.to_csv('./file_list_all.csv') 





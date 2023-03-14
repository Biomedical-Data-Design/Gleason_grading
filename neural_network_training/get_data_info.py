import pandas as pd
import numpy as np
file = pd.read_csv("./file_list_subset1.txt")
file_list = file["img"]
n = 0
label_list = []
for i in file_list:
    feature = np.load(str("/home/yzhao155/data-acharl15/gleason_grading/pretrain/"+i.split('.tiff')[0]+"/feature.npy"))
    label = np.load(str("/home/yzhao155/data-acharl15/gleason_grading/pretrain/"+i.split('.tiff')[0]+"/class.npy"))
    #print(feature.shape,label.shape)
    for j in label:
        label_list.append(j)
    n+=len(label)
print(n)
print(label_list[0:10])
label_list = np.array(label_list)
print(len(label_list[label_list== 0]), sum(label_list[label_list== 1]),sum(label_list[label_list== 2]),sum(label_list[label_list== 3]),sum(label_list[label_list== 4]))



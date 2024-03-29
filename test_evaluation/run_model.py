from scipy.special import softmax
import torch
from torchvision import transforms
import sys 
sys.path.append("../neural_network_training")
from Model.model import Model_name
from mtdp import build_model
from collections import OrderedDict
from Dataset.dataset import Dataset_Image
import torch.utils.data as data_utils
import torch.nn as nn
import numpy as np
import pickle
import os
import cv2
from PIL import Image

# f = args.filename
# print(f)

def heatmap(c0,c1,c2,c3,c4):
    c0 = [c0* c for c in [0, 255, 0]]
    c1 = [c1* c for c in [255, 0, 0]]
    c2 = [c2* c for c in [0,255,255]]
    c3 = [c3* c for c in [255,0,255]]
    c4 = [c4* c for c in [0, 0,255]]
    pixel = [sum(x) for x in zip(c0,c1,c2,c3,c4)]
    return pixel


def Evaluate(param,out_dir,patch_folder,image):
    ################### define model
    # define a device which allows you to use GPU resources
    device = torch.device("cuda")
    # define your model
    pretrained_model = build_model(arch="densenet121", pretrained="mtdp", pool= True)
    model = Model_name(feature_encoder = pretrained_model)

    #model = torch.nn.DataParallel(model)
    param=torch.load(param)
    #print(param)
    model_dict = model.state_dict()
    pretrained_dict = {k.partition('module.')[2]: v for k, v in param.items() if k.partition('module.')[2] in model_dict}
    #print(pretrained_dict)
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    print("loaded model")
    ### utilize GPU
    model = model.to(device)
    print("model_inputed")

    m = nn.Softmax(dim=1).to(device)
    ################# load data
    #out_dir = str("/home/yzhao155/data-acharl15/gleason_grading/test_folder/result/"+f+"/")
    isExist = os.path.exists(out_dir)
    if not isExist:
        print("new_directory",out_dir)
        os.makedirs(out_dir)
    print(out_dir)
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ])


    img = Dataset_Image(str(patch_folder+"patch_mask.jpg"), 
                        image,transform = transform)
    #print(img)
    train_loader = data_utils.DataLoader(img, batch_size=32, shuffle=False)

    print("load img")

    with torch.no_grad():
        model.eval()
        index_list_x = []
        index_list_y = []
        yprob = []
        for i, (x, idx) in enumerate(train_loader):
            #print("> train iter #{}".format(i + 1))
            
            out = model(x.to(device))
            ypred = m(out)
            #print(ypred.shape)
            #print("out",out,"label",y)
            for j in range(ypred.shape[0]):
                index_list_x.append(idx[0][j])
                index_list_y.append(idx[1][j])
                yprob.append(ypred.cpu().detach().numpy()[j])

    yprob= np.array(yprob)
    index_list_x = np.array(index_list_x)
    index_list_y = np.array(index_list_y)
    print(yprob.shape)
    ###########save result
    output=dict()
    output["yprob"] = yprob
    output["index_x"] = index_list_x
    output["index_y"] = index_list_y

    #out_dir
    #print(output)
    print('\nSaving the object')
    with open(str(out_dir+"result.pck"), "wb") as output_file:
        pickle.dump(output, output_file)
    ##########export result
    # heatmap H*W*5
    H = np.zeros((np.max(index_list_x)+1,np.max(index_list_y)+1,5))
    G_pred =  np.zeros((np.max(index_list_x)+1,np.max(index_list_y)+1))
    G_pred_color =  np.zeros((np.max(index_list_x)+1,np.max(index_list_y)+1,3))
    print(H.shape)
    for i in range(yprob.shape[0]):
        x = index_list_x[i]
        y = index_list_y[i]
        # heatmap
        H[x,y,0] = yprob[i][0]
        H[x,y,1] = yprob[i][1]
        H[x,y,2] = yprob[i][2]
        H[x,y,3] = yprob[i][3]
        H[x,y,4] = yprob[i][4]

        #Gpred
        #green=normal(0, 255, 0), blue=stroma (255, 0, 0), yellow=3 (0,255,255), fuchsia=4 (255,0,255), red=5 (0, 0,255)
        # alpha 
        G_pred_color[x,y] = heatmap(yprob[i][0],yprob[i][1],yprob[i][2],yprob[i][3],yprob[i][4])
        
        G_pred[x,y] = np.argmax(yprob[i])+1


    #create black and white classification masks
    # H = cv2.convertScaleAbs(H, alpha=(255.0))
    # cv2.imwrite(str(out_dir+"heatmap.jpg"), H)
    
    ##np.save(str(out_dir+"H"), H)
    ##np.save(str(out_dir+"G_pred"), G_pred)

    cv2.imwrite(str(out_dir+"G_pred_color.jpg"), G_pred_color)
    


    G_pred = ((G_pred - G_pred.min()) * (1/(5 - G_pred.min()) * 255)).astype('uint8')
    kernel = np.ones((5,5),np.uint8)
    #G_pred = cv2.morphologyEx(G_pred, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(str(out_dir+"G_pred.jpg"), G_pred)
    print("result output")
    for i in range(5):
        tmp_h = H[:,:,i]
        tmp_h = ((tmp_h - tmp_h.min()) * (1/(1 - tmp_h.min()) * 255)).astype('uint8')
        tmp_h = cv2.morphologyEx(tmp_h, cv2.MORPH_CLOSE, kernel)
        H[:,:,i] = tmp_h
        ##cv2.imwrite(str(out_dir+"G_pred_class"+str(i)+".jpg"), tmp_h)

    G_pred_after_mor =  np.zeros((np.max(index_list_x)+1,np.max(index_list_y)+1))
    G_pred_after_mor_color =  np.zeros((np.max(index_list_x)+1,np.max(index_list_y)+1,3))
    for i in range(len(yprob)):
        x = index_list_x[i]
        y = index_list_y[i]
        G_pred_after_mor[x,y] = np.argmax(H[x,y])+1
        tmp = softmax(H[x,y])
        #print(np.sum(tmp))
        G_pred_after_mor_color[x,y] = heatmap(tmp[0],tmp[1],tmp[2],tmp[3],tmp[4])

    ##np.save(str(out_dir+"G_pred_after_mor"), G_pred_after_mor)
    G_pred_after_mor = ((G_pred_after_mor - G_pred_after_mor.min()) * (1/(5 - G_pred_after_mor.min()) * 255)).astype('uint8')
    cv2.imwrite(str(out_dir+"G_pred_after_mor.jpg"), G_pred_after_mor)
    cv2.imwrite(str(out_dir+"G_pred_after_mor_color.jpg"), G_pred_after_mor_color)



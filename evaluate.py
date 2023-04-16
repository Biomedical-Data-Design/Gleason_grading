import torch
from torchvision import transforms
from Model.model import Model_name
from mtdp import build_model
from collections import OrderedDict
from Dataset.dataset import Dataset_name
import torch.utils.data as data_utils
import torch.nn as nn
import numpy as np
import pickle
import os
import cv2
from PIL import Image

# f = args.filename
# print(f)
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


    img = Dataset_name(str(patch_folder+"patch_mask.jpg"), 
                        str(patch_folder+"patch_mask_"),
                        image,transform = transform)
    #print(img)
    train_loader = data_utils.DataLoader(img, batch_size=32, shuffle=False)

    print("load img")

    with torch.no_grad():
        model.eval()
        # train
        label = []
        index_list_x = []
        index_list_y = []
        yprob = []
        for i, (x, idx,y) in enumerate(train_loader):
            #print("> train iter #{}".format(i + 1))
            
            out = model(x.to(device))
            ypred = m(out)
            #print("out",out,"label",y)
            for j in range(len(y)):
                index_list_x.append(idx[0][j])
                index_list_y.append(idx[1][j])
                label.append(y[j])
                yprob.append(ypred.cpu().detach().numpy()[j])

    label = np.array(label)
    yprob= np.array(yprob)
    index_list_x = np.array(index_list_x)
    index_list_y = np.array(index_list_y)
    print(label.shape,yprob.shape,)
    ###########save result
    output=dict()
    output["yprob"] = yprob
    output["ytrue"] = label
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
    G = np.zeros((np.max(index_list_x)+1,np.max(index_list_y)+1))
    G_pred =  np.zeros((np.max(index_list_x)+1,np.max(index_list_y)+1))
    print(H.shape)
    for i in range(len(label)):
        x = index_list_x[i]
        y = index_list_y[i]

        # heatmap
        H[x,y,0] = yprob[i][0]
        H[x,y,1] = yprob[i][1]
        H[x,y,2] = yprob[i][2]
        H[x,y,3] = yprob[i][3]
        H[x,y,4] = yprob[i][4]

        #Gpred
        G_pred[x,y] = np.argmax(yprob[i])+1

        #Ground truth
        #print(i,x,y)
        G[x,y] = label[i]+1
        #print(np.argmax(yprob[i]),label[i])

    #G_pred = np.argmax(H, axis = -1) + 1
    #print(G_pred)
    #print(G)
    print("max",np.max(G_pred))

    #create black and white classification masks
    # H = cv2.convertScaleAbs(H, alpha=(255.0))
    # cv2.imwrite(str(out_dir+"heatmap.jpg"), H)
    np.save(str(out_dir+"H"), H)

    #G = cv2.convertScaleAbs(G, alpha=(255.0))
    #cv2.imwrite(str(out_dir+"Ground_truth.jpg"), G)
    G = ((G - G.min()) * (1/(5 - G.min()) * 255)).astype('uint8')
    #G.save(str(out_dir+"Ground_truth.jpg"))
    cv2.imwrite(str(out_dir+"Ground_truth.jpg"), G)
    np.save(str(out_dir+"G"), G)

    #G_pred = cv2.convertScaleAbs(G_pred, alpha=(255.0))
    #cv2.imwrite(str(out_dir+"G_pred.jpg"), G_pred)
    G_pred = ((G_pred - G_pred.min()) * (1/(5 - G_pred.min()) * 255)).astype('uint8')
    cv2.imwrite(str(out_dir+"G_pred.jpg"), G_pred)
    np.save(str(out_dir+"G_pred"), G_pred)
    print("result output")


    for i in range(5):
        tmp_h = H[:,:,i]
        tmp_h = ((tmp_h - tmp_h.min()) * (1/(1 - tmp_h.min()) * 255)).astype('uint8')
        cv2.imwrite(str(out_dir+"G_pred_class"+str(i)+".jpg"), tmp_h)

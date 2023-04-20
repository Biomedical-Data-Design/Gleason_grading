import os
import numpy as np
from f1_eval import aggc_eval,total_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,balanced_accuracy_score

def get_pred_truth(prefix):
    G_pred = []
    G_truth = []
    G_pred_after_mor = []

    for i in os.listdir("/data/acharl15/gleason_grading/test_folder/result/"):
        if i.startswith(prefix):
            if os.path.exists(str("/data/acharl15/gleason_grading/test_folder/result/"+i+'/G_pred.npy')):
                gpred = np.load(str("/data/acharl15/gleason_grading/test_folder/result/"+i+'/G_pred.npy'))
                gpred_mor = np.load(str("/data/acharl15/gleason_grading/test_folder/result/"+i+'/G_pred_after_mor.npy'))
                gtruth = np.load(str("/data/acharl15/gleason_grading/test_folder/result/"+i+'/G.npy'))
                for x in range(gpred.shape[0]):
                    for y in range(gpred.shape[1]):
                        G_pred.append(gpred[x,y])
                        G_truth.append(gtruth[x,y])
                        G_pred_after_mor.append(gpred_mor[x,y])
            else:
                print(i)

    G_truth = np.array(G_truth)
    G_pred_after_mor = np.array(G_pred_after_mor)
    G_pred = np.array(G_pred)
    idx = np.where(G_truth != 0)
    G_truth = G_truth[idx]
    G_pred_after_mor = G_pred_after_mor[idx]
    G_pred = G_pred[idx]

    return G_pred,G_truth,G_pred_after_mor

s1pred,s1truth,s1predafter = get_pred_truth("Subset1")
s2pred,s2truth,s2predafter = get_pred_truth("Subset2")
s3pred,s3truth,s3predafter = get_pred_truth("Subset3")

s1= aggc_eval(s1truth, s1pred)
s1mor= aggc_eval(s1truth, s1predafter)
s2= aggc_eval(s2truth, s2pred)
s2mor= aggc_eval(s2truth, s2predafter)
s3= aggc_eval(s3truth, s3pred)
s3mor= aggc_eval(s3truth, s3predafter)

print("Subset1 raw_score",s1)
print("Subset1 f1_score_after_morphological_transforamtion",s1mor)
print("Subset2 raw_score",s2)
print("Subset2 f1_score_after_morphological_transforamtion",s2mor)
print("Subset3 raw_score",s3)
print("Subset3 f1_score_after_morphological_transforamtion",s3mor)

print("total",total_score(s1,s2,s3))
print("total f1_score_after_morphological_transforamtion",total_score(s1mor,s2mor,s3mor))


#### accuracy
def cal_acc(ytrue,ypred):
    return(accuracy_score(ytrue,ypred),balanced_accuracy_score(ytrue,ypred))

s1pred,s1truth,s1predafter = get_pred_truth("Subset1")
s2pred,s2truth,s2predafter = get_pred_truth("Subset2")
s3pred,s3truth,s3predafter = get_pred_truth("Subset3")

s1acc = [accuracy_score(s1truth,s1pred),balanced_accuracy_score(s1truth,s1pred)]
s1accmore = [accuracy_score(s1truth,s1predafter),balanced_accuracy_score(s1truth,s1predafter)]

s2acc = [accuracy_score(s2truth,s2pred),balanced_accuracy_score(s2truth,s2pred)]
s2accmore = [accuracy_score(s2truth,s2predafter),balanced_accuracy_score(s2truth,s2predafter)]

s3acc = [accuracy_score(s3truth,s3pred),balanced_accuracy_score(s3truth,s3pred)]
s3accmore = [accuracy_score(s3truth,s3predafter),balanced_accuracy_score(s3truth,s3predafter)]

print("Accuracy")
print("Subset1 raw_score",s1acc)
print("Subset1 acc_score_after_morphological_transforamtion",s1accmore)
print("Subset2 raw_score",s2acc)
print("Subset2 acc_score_after_morphological_transforamtion",s2accmore)
print("Subset3 raw_score",s3acc)
print("Subset3 acc_score_after_morphological_transforamtion",s3accmore)

print("total",(s1acc[0]+s2acc[0]+s3acc[0])/3,(s1acc[1]+s2acc[1]+s3acc[1])/3)
print("total acc_score_after_morphological_transforamtion",(s1accmore[0]+s2accmore[0]+s3accmore[0])/3,(s1accmore[1]+s2accmore[1]+s3accmore[1])/3)
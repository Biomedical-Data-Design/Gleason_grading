#function aggc_eval gives the weighted score for an individual subset
#function total_score gives the total weighted score

#inputs: true and predicted labels for a single subset
#outputs: weighted f1 score
import numpy as np

def aggc_eval(true, pred):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score


    # calculate confusion matrix for each subset
    conf_mat=np.zeros((6,6))
    conf_mat[1:6,1:6] = confusion_matrix(true, pred)


    # Weighted-average F1-score = 0.25 * F1-score_G3 + 0.25 * F1-score_G4 +0.25 * F1-score_G5 +0.125 * F1-score_Normal +0.125 * F1-score_Stroma, where:
    #
    # F1-score=2×Precision×Recall/(Precision+ Recall);Precision=TP/(TP+FP);Recall=TP/(TP+FN)

    Stroma_Recall = conf_mat[1, 1] / np.sum(conf_mat[1, :])
    Normal_Recall = conf_mat[2, 2] / np.sum(conf_mat[2, :])
    G3_Recall = conf_mat[3, 3] / np.sum(conf_mat[3, :])
    G4_Recall = conf_mat[4, 4] / np.sum(conf_mat[4, :])
    G5_Recall = conf_mat[5, 5] / np.sum(conf_mat[5, :])

    Stroma_Pre = conf_mat[1, 1] / (np.sum(conf_mat[:, 1]) - conf_mat[0, 1])
    Normal_Pre = conf_mat[2, 2] / (np.sum(conf_mat[:, 2]) - conf_mat[0, 2])
    G3_Pre = conf_mat[3, 3] / (np.sum(conf_mat[:, 3]) - conf_mat[0, 3])
    G4_Pre = conf_mat[4, 4] / (np.sum(conf_mat[:, 4]) - conf_mat[0, 4])
    G5_Pre = conf_mat[5, 5] / (np.sum(conf_mat[:, 5]) - conf_mat[0, 5])

    F1_Stroma = 2 * Stroma_Pre * Stroma_Recall / (Stroma_Pre + Stroma_Recall)
    F1_Normal = 2 * Normal_Pre * Normal_Recall / (Normal_Pre + Normal_Recall)
    F1_G3 = 2 * G3_Pre * G3_Recall / (G3_Pre + G3_Recall)
    F1_G4 = 2 * G4_Pre * G4_Recall / (G4_Pre + G4_Recall)
    F1_G5 = 2 * G5_Pre * G5_Recall / (G5_Pre + G5_Recall)

    weighted = 0.25 * F1_G3 + 0.25 * F1_G4 + 0.25 * F1_G5 + 0.125 * F1_Normal + 0.125 * F1_Stroma



    return weighted


#inputs: weighted scores for all 3 subsets, outputs: total weighted score
def total_score(s1_weighted, s2_weighted, s3_weighted):

    return 0.6*s1_weighted + 0.2*s2_weighted + 0.2*s3_weighted

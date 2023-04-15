#inputs: true and predicted labels for each subset
#outputs:
    #f1 score for each subset (s1_score, s2_score, s3_score),
    #weighted f1 score for each subset (weighted_s1, weighted_s2, weighted_s3)
    #total weighted f1 score (total_score)
import numpy as np

def aggc_eval(s1_true, s1_pred, s2_true, s2_pred, s3_true, s3_pred):
    from sklearn.metrics import f1_score

    #calculate overall f1 score for each subset
    s1_score=f1_score(s1_true, s1_pred, average="micro")
    s2_score=f1_score(s2_true,s2_pred, average="micro")
    s3_score=f1_score(s3_true,s3_pred, average="micro")

    # slice by grade --------------------------------
    #subset 1
    s1 = np.vstack((s1_true, s1_pred))

    slice1_1 = s1[:, (s1_true == 1)]
    slice2_1 = s1[:, (s1_true == 2)]
    slice3_1 = s1[:, (s1_true == 3)]
    slice4_1 = s1[:, (s1_true == 4)]
    slice5_1 = s1[:, (s1_true == 5)]

    #subset 2
    s2 = np.vstack((s2_true, s2_pred))

    slice1_2 = s2[:, (s2_true == 1)]
    slice2_2 = s2[:, (s2_true == 2)]
    slice3_2 = s2[:, (s2_true == 3)]
    slice4_2 = s2[:, (s2_true == 4)]
    slice5_2 = s2[:, (s2_true == 5)]

    #subet 3
    s3 = np.vstack((s3_true, s3_pred))

    slice1_3 = s3[:, (s3_true == 1)]
    slice2_3 = s3[:, (s3_true == 2)]
    slice3_3 = s3[:, (s3_true == 3)]
    slice4_3 = s3[:, (s3_true == 4)]
    slice5_3 = s3[:, (s3_true == 5)]
    #-----------------------------------------------


    #apply evaluation weightings as prescribed by aggc

    weighted_s1 = 0.25 * (f1_score(slice3_1[0, :], slice3_1[1, :], average="micro")) + 0.25 * (
        f1_score(slice4_1[0, :], slice4_1[1, :], average="micro")) + 0.25 * (f1_score(slice5_1[0, :], slice5_1[1, :], average="micro")) + 0.125 * (
                      f1_score(slice1_1[0, :], slice1_1[1, :], average="micro")) + 0.125 * (f1_score(slice2_1[0, :], slice2_1[1, :], average="micro"))

    weighted_s2 = 0.25 * (f1_score(slice3_2[0, :], slice3_2[1, :], average="micro")) + 0.25 * (
        f1_score(slice4_2[0, :], slice4_2[1, :], average="micro")) + 0.25 * (f1_score(slice5_2[0, :], slice5_2[1, :], average="micro")) + 0.125 * (
                      f1_score(slice1_2[0, :], slice1_2[1, :], average="micro")) + 0.125 * (f1_score(slice2_2[0, :], slice2_2[1, :], average="micro"))

    weighted_s3 = 0.25 * (f1_score(slice3_3[0, :], slice3_3[1, :], average="micro")) + 0.25 * (
        f1_score(slice4_3[0, :], slice4_3[1, :], average="micro")) + 0.25 * (f1_score(slice5_3[0, :], slice5_3[1, :], average="micro")) + 0.125 * (
                      f1_score(slice1_3[0, :], slice1_3[1, :], average="micro")) + 0.125 * (f1_score(slice2_3[0, :], slice2_3[1, :], average="micro"))


    total_score = 0.6*weighted_s1 + 0.2*weighted_s3 + 0.2*weighted_s2


    return s1_score, s2_score, s3_score, weighted_s1, weighted_s2, weighted_s3, total_score

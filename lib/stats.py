import numpy as np
import pandas as pd
from scipy import stats

def sigmoid(x): # Sigmoid function
    return 1 / (1 + np.exp(-x))

def ICC(cse, typ, dat):
    """
    Adapted from Matlab function in https://github.com/ZhiwenShao/ARL
    Compute Intraclass Correlation Coefficients (ICC)

    Parameters:
        cse (int): 1, 2, or 3 depending on the raters configuration
        typ (str): 'single' or 'k' based on whether the ICC is based on a single measurement or on an average
        dat (array): data with raters/ratings in rows and targets in columns
    """

    # number of raters/ratings
    k = dat.shape[1]
    # number of targets
    n = dat.shape[0]
    # mean per target
    mpt = np.mean(dat, axis=1)
    # mean per rater/rating
    mpr = np.mean(dat, axis=0)
    # get total mean
    tm = np.mean(mpt)
    # within target sum squares
    WSS = np.sum((dat - mpt[:, np.newaxis])**2)
    # within target mean squares
    WMS = WSS / (n * (k - 1))
    # between rater sum squares
    RSS = np.sum((mpr - tm)**2) * n
    # between rater mean squares
    RMS = RSS / (k - 1)
    # between target sum squares
    BSS = np.sum((mpt - tm)**2) * k
    # between targets mean squares
    BMS = BSS / (n - 1)
    # residual sum of squares
    ESS = WSS - RSS
    # residual mean squares
    EMS = ESS / ((k - 1) * (n - 1))

    if cse == 1:
        if typ == 'single':
            return (BMS - WMS) / (BMS + (k - 1) * WMS)
        elif typ == 'k':
            return (BMS - WMS) / BMS
        else:
            raise ValueError('Wrong value for input typ')
    elif cse == 2:
        if typ == 'single':
            return (BMS - EMS) / (BMS + (k - 1) * EMS + k * (RMS - EMS) / n)
        elif typ == 'k':
            return (BMS - EMS) / (BMS + (RMS - EMS) / n)
        else:
            raise ValueError('Wrong value for input typ')
    elif cse == 3:
        if typ == 'single':
            return (BMS - EMS) / (BMS + (k - 1) * EMS)
        elif typ == 'k':
            return (BMS - EMS) / BMS
        else:
            raise ValueError('Wrong value for input typ')
    else:
        raise ValueError('Wrong value for input cse')

def compute_icc_3_1(pred_labels, true_labels): # Compute ICC(3,1)
    n_AUs = pred_labels.shape[1]

    icc_3_1_values = []

    for au in range(n_AUs):
        if np.all(true_labels[:, au] == -1):
            icc_3_1_values.append(float('nan'))
        else:
            icc_3_1_values.append(ICC(3, 'single', np.hstack((pred_labels[:,au:(au+1)], true_labels[:,au:(au+1)]))))
    return icc_3_1_values

def compute_f1_score(pred_labels, true_labels): # Compute the F1 score and output the precision and recall as well
    TP = np.sum((true_labels == True) & (pred_labels == True))
    FP = np.sum((true_labels == False) & (pred_labels == True))
    FN = np.sum((true_labels == True) & (pred_labels == False))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    if precision + recall == 0:
        return 0.0, precision, recall
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1, precision, recall

def compute_results_for_intensity_estimation(info, all_pred_labels, all_true_labels, subject_indices, sub_IDs): # Compute and print the evaluation metrics for AU intensity estimation
    PCCs = {}
    acrossICCs = {}
    withinICCs = {}
    MAEs = {}
    MSEs = {}
    for i, AU in enumerate(info["AUs"]):
        valid_idx = np.where(all_true_labels[:,i] >= 0)[0]
        PCCs[AU] = stats.pearsonr(all_pred_labels[valid_idx,i], all_true_labels[valid_idx,i]).statistic
        acrossICCs[AU] = compute_icc_3_1(all_pred_labels[valid_idx,i:(i+1)], all_true_labels[valid_idx,i:(i+1)])[0]
        withinICCs[AU] = np.mean([compute_icc_3_1(all_pred_labels[np.intersect1d(valid_idx, subject_indices[sub_ID]),i:(i+1)], all_true_labels[np.intersect1d(valid_idx, subject_indices[sub_ID]),i:(i+1)])[0] for sub_ID in sub_IDs])
        MAEs[AU] = np.mean(np.abs(all_pred_labels[valid_idx,i] - all_true_labels[valid_idx,i]))
        MSEs[AU] = np.mean(np.power(all_pred_labels[valid_idx,i] - all_true_labels[valid_idx,i], 2))
    PCCs_with_avg = np.array(list(PCCs.values()) + [np.mean(list(PCCs.values()))]).round(2)
    acrossICCs_with_avg = np.array(list(acrossICCs.values()) + [np.mean(list(acrossICCs.values()))]).round(2)
    withinICCs_with_avg = np.array(list(withinICCs.values()) + [np.mean(list(withinICCs.values()))]).round(2)
    MAEs_with_avg = np.array(list(MAEs.values()) + [np.mean(list(MAEs.values()))]).round(2)
    MSEs_with_avg = np.array(list(MSEs.values()) + [np.mean(list(MSEs.values()))]).round(2)
    print(f"PCCs: {' & '.join(list(map(lambda x: '{:.2f}'.format(x)[1:], PCCs_with_avg)))}")
    print(f"acrossICCs: {' & '.join(list(map(lambda x: '{:.2f}'.format(x)[1:], acrossICCs_with_avg)))}")
    print(f"withinICCs: {' & '.join(list(map(lambda x: '{:.2f}'.format(x)[1:], withinICCs_with_avg)))}")
    print(f"MAEs: {' & '.join(list(map(lambda x: '{:.2f}'.format(x)[1:], MAEs_with_avg)))}")
    print(f"MSEs: {' & '.join(list(map(lambda x: '{:.2f}'.format(x)[1:], MSEs_with_avg)))}")

    df = pd.DataFrame({'AU': info["AUs"] + ['average'],
                       'PCC': PCCs_with_avg,
                       'across-participant ICC(3,1)': acrossICCs_with_avg,
                       'within-participant ICC(3,1)': withinICCs_with_avg,
                       'MAE': MAEs_with_avg,
                       'MSE': MSEs_with_avg})
    df = df.set_index('AU')
    return df

def compute_results_for_occurrence_detection(info, all_pred_labels, all_true_labels, subject_indices): # Compute and print the evaluation metrics for AU detection
    accs = {}
    F1_scores = {}
    precisions = {}
    recalls = {}
    for i, AU in enumerate(info["AUs"]):
        valid_idx = np.where(all_true_labels[:,i] >= 0)[0]
        accs[AU] = np.mean((all_pred_labels[valid_idx,i] >= 0.0) == all_true_labels[valid_idx,i], axis=0)
        F1_scores[AU], precisions[AU], recalls[AU] = compute_f1_score(all_pred_labels[valid_idx,i] >= 0.0, all_true_labels[valid_idx,i])
    accs_with_avg = list(accs.values()) + [np.mean(list(accs.values()))]
    F1_scores_with_avg = list(F1_scores.values()) + [np.mean(list(F1_scores.values()))]
    precisions_with_avg = list(precisions.values()) + [np.mean(list(precisions.values()))]
    recalls_with_avg = list(recalls.values()) + [np.mean(list(recalls.values()))]
    print(f"accs: {' & '.join(list(map(lambda x: '{:.1f}'.format(x*100), accs_with_avg)))}")
    print(f"F1_scores: {' & '.join(list(map(lambda x: '{:.1f}'.format(x*100), F1_scores_with_avg)))}")
    print(f"precisions: {' & '.join(list(map(lambda x: '{:.1f}'.format(x*100), precisions_with_avg)))}")
    print(f"recalls: {' & '.join(list(map(lambda x: '{:.1f}'.format(x*100), recalls_with_avg)))}")

    df = pd.DataFrame({'AU': info["AUs"] + ['average'],
                       'Accuracy': np.array(accs_with_avg).round(3) * 100,
                       'F1 score': np.array(F1_scores_with_avg).round(3) * 100,
                       'Precision': np.array(precisions_with_avg).round(3) * 100,
                       'Recall': np.array(recalls_with_avg).round(3) * 100})
    df = df.set_index('AU')
    return df
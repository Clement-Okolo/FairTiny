import os
import numpy as np
import random
import pandas as pd
from sklearn.metrics import *
import scipy.stats as stats 
from sklearn.utils import *

import torch

def compute_auc(pred_prob, y, num_classes=2):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    if num_classes == 2:
        fpr, tpr, thresholds = roc_curve(y, pred_prob)
        auc_val = auc(fpr, tpr)
    elif num_classes > 2:
        y_onehot = num_to_onehot(y, num_classes)
        auc_val = roc_auc_score(y_onehot, pred_prob, average='macro', multi_class='ovr')

    return auc_val

def bootstrap(preds, gts, repeat_times=10, random_state=0, n_samples=100):
    bootstrap_aucs = []
    for i in range(repeat_times):
        # # Sample size for bootstrapping
        # sample_size = len(preds)

        # # Generating random indices for bootstrapping
        # random_indices = np.random.choice(len(preds), size=sample_size, replace=True)

        # # Extracting pairs using the generated indices
        # bootstrap_pairs_array1 = preds[random_indices]
        # bootstrap_pairs_array2 = gts[random_indices]

        bootstrap_pairs_array1, bootstrap_pairs_array2 = resample(preds, gts, replace=True, n_samples=n_samples) #, n_samples=sample_size, random_state=random_state

        overall_auc = compute_auc(bootstrap_pairs_array1, bootstrap_pairs_array2, num_classes=2)
        bootstrap_aucs.append(overall_auc)

    return bootstrap_aucs


input_npz_1 = r'/your/path/pred_gt_best_epoch.npz'            # fairTiny-student
input_npz_2 = r'/your/path/mobilenet-pred_gt_best_epoch.npz'  # baseline


repeat_times = [100, 1000, 2000, 3000, 4000, 5000]
# n_samples = 2000

raw_data = np.load(input_npz_1)

# ['val_pred', 'val_gt', 'val_attr']
# ['test_pred', 'test_gt', 'test_attr']
preds_1 = raw_data['test_pred']
gts_1 = raw_data['test_gt']
attrs_1 = raw_data['test_attr'] if 'test_attr' in raw_data else None

raw_data = np.load(input_npz_2)
preds_2 = raw_data['test_pred']
gts_2 = raw_data['test_gt']
attrs_2 = raw_data['test_attr'] if 'test_attr' in raw_data else None

random_state = random.randint(0, 1e+6)

print(f'random state: {random_state}')

for i in repeat_times:
    # Overall
    auc_1st_npz = bootstrap(preds_1, gts_1, repeat_times=i, n_samples=len(preds_1))
    auc_2nd_npz = bootstrap(preds_2, gts_2, repeat_times=i, n_samples=len(preds_1))
    result = stats.ttest_rel(auc_1st_npz, auc_2nd_npz)
    print(f'Overall: repeat {i} times, p-value: {result.pvalue}')

    # Per race (row 0) [0: Asian, 1: Black, 2: White]
    if attrs_1 is not None and attrs_2 is not None:
        for race in [0, 1, 2]:
            idx_1 = np.where(attrs_1[0] == race)[0]
            idx_2 = np.where(attrs_2[0] == race)[0]
            group_name = f'Race {race}'
            if len(idx_1) > 0 and len(idx_2) > 0:
                auc_1 = bootstrap(preds_1[idx_1], gts_1[idx_1], repeat_times=i, n_samples=len(idx_1))
                auc_2 = bootstrap(preds_2[idx_2], gts_2[idx_2], repeat_times=i, n_samples=len(idx_2))
                res = stats.ttest_rel(auc_1, auc_2)
                print(f'{group_name}: repeat {i} times, p-value: {res.pvalue:} (n1={len(idx_1)}, n2={len(idx_2)})')
            else:
                print(f'{group_name}: repeat {i} times, Not enough samples (n1={len(idx_1)}, n2={len(idx_2)})')
        
        # Per gender (row 1) [0: female, 1: male])
        for gender in [0, 1]:
            idx_1 = np.where(attrs_1[1] == gender)[0]
            idx_2 = np.where(attrs_2[1] == gender)[0]
            group_name = f'Gender {gender}'
            if len(idx_1) > 0 and len(idx_2) > 0:
                auc_1 = bootstrap(preds_1[idx_1], gts_1[idx_1], repeat_times=i, n_samples=len(idx_1))
                auc_2 = bootstrap(preds_2[idx_2], gts_2[idx_2], repeat_times=i, n_samples=len(idx_2))
                res = stats.ttest_rel(auc_1, auc_2)
                print(f'{group_name}: repeat {i} times, p-value: {res.pvalue} (n1={len(idx_1)}, n2={len(idx_2)})')
            else:
                print(f'{group_name}: repeat {i} times, Not enough samples (n1={len(idx_1)}, n2={len(idx_2)})')

        # Per ethnicity (row 2, [0: non-hispanic, 1:hispanic])
        for eth in [0, 1]:
            idx_1 = np.where(attrs_1[2] == eth)[0]
            idx_2 = np.where(attrs_2[2] == eth)[0]
            group_name = f'Ethnicity {eth}'
            if len(idx_1) > 0 and len(idx_2) > 0:
                auc_1 = bootstrap(preds_1[idx_1], gts_1[idx_1], repeat_times=i, n_samples=len(idx_1))
                auc_2 = bootstrap(preds_2[idx_2], gts_2[idx_2], repeat_times=i, n_samples=len(idx_2))
                res = stats.ttest_rel(auc_1, auc_2)
                print(f'{group_name}: repeat {i} times, p-value: {res.pvalue} (n1={len(idx_1)}, n2={len(idx_2)})')
            else:
                print(f'{group_name}: repeat {i} times, Not enough samples (n1={len(idx_1)}, n2={len(idx_2)})')

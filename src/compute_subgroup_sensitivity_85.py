import numpy as np

# input_npz_1 = (convnext-t_pred_gt_best_fair_distill_epoch.npz, 
#                    swin-t_pred_gt_best_fair_distill_epoch.npz,
#                         resnet-pred_gt_best_fair_distill_epoch.npz,
#                            densnet-pred_gt_best_fair_distill_epoch.npz,
#                                deit-small-pred_gt_best_fair_distill_epoch.npz,
#                                    mobilenet-pred_gt_best_fair_distill_epoch.npz,
#                                        efficientnet-pred_gt_best_fair_distill_epoch.npz)

# input_npz_2 = (convnext_slo_fundus_lr1e-4_bz10_seed13_auc0.7848, 
#                  swin_t_slo_fundus_lr1e-4_bz32_seed13_auc0.7977, 
#                   deit_small_slo_fundus_lr1e-4_bz32_seed13_auc0.7761
#                    ViT-B_slo_fundus_lr1e-4_bz64_seed13_auc0.8480
#                        resnet-pred_gt_best_epoch.npz
#                            efficientnet-pred_gt_best_epoch.npz
#                                densenet-pred_gt_best_epoch.npz
#                                    mobilenet-pred_gt_best_epoch.npz)

# input_npz_3 = ( convnext_slo_fundus_lr1e-4_bz10_seed13_auc0.7813, 
#                    resnet_slo_fundus_lr1e-4_bz10_seed13_auc0.7481,
#                      deit_small_slo_fundus_lr1e-4_bz32_seed13_auc0.7204  
#                         swin_t_slo_fundus_lr1e-4_bz32_seed13_auc0.7720
#                           densenet_slo_fundus_lr5e-4_bz64_seed13_auc0.7542
#                             efficientnet_slo_fundus_lr1e-4_bz10_seed13_auc0.7965
#                                mobilenet_slo_fundus_lr1e-4_bz10_seed13_auc0.7769 )

# input_npz_1 = '/medailab/medailab/clement/FairAdaptiveScaling/scripts/fair_npz_files/resnet-pred_gt_best_fair_distill_epoch.npz'
# input_npz_2 = '/medailab/medailab/clement/FairAdaptiveScaling/scripts/results_harvard10k/dr_slo_fundus_race/efficientnet-pred_gt_best_epoch.npz'
# input_npz_3 = '/results_harvard10k/oversampling/dr_slo_fundus_race//pred_gt_best_epoch.npz'
# input_npz_4 = /results_harvard10k/dr_slo_fundus_adversarial/dr_slo_fundus_race/mobilenet_slo_fundus_lr1e-4_bz10_adv_seed13_auc0.7533/

# /dr_slo_fundus_adversarial/dr_slo_fundus_race/densenet_slo_fundus_lr5e-4_bz64_adv_seed13_auc0.7928/
# /dr_slo_fundus_adversarial/dr_slo_fundus_race/efficientnet_slo_fundus_lr1e-4_bz32_adv_seed13_auc0.7847/
# /dr_slo_fundus_adversarial/dr_slo_fundus_race/deit_small_slo_fundus_lr1e-4_bz32_adv_seed13_auc0.7727/
# /dr_slo_fundus_adversarial/dr_slo_fundus_race/swin_t_slo_fundus_lr1e-4_bz32_adv_seed13_auc0.7926/
# /dr_slo_fundus_adversarial/dr_slo_fundus_race/convnext_slo_fundus_lr1e-4_bz10_adv_seed13_auc0.7836/
# /dr_slo_fundus_adversarial/dr_slo_fundus_race/resnet_slo_fundus_lr5e-4_bz10_adv_seed13_auc0.6699/

# /dr_slo_fundus_FAS/dr_slo_fundus_race_fAS/mobilenet_slo_fundus_lr1e-4_bz10_seed13_auc0.7682/
# /dr_slo_fundus_FAS/dr_slo_fundus_race_fAS/densenet_slo_fundus_lr5e-4_bz64_seed13_auc0.7871/
# /dr_slo_fundus_FAS/dr_slo_fundus_race_fAS/efficientnet_slo_fundus_lr1e-4_bz32_seed13_auc0.7900/
# /dr_slo_fundus_FAS/dr_slo_fundus_race_fAS/deit_small_slo_fundus_lr1e-4_bz32_seed13_auc0.7560/
# /dr_slo_fundus_FAS/dr_slo_fundus_race_fAS/swin_t_slo_fundus_lr1e-4_bz32_seed13_auc0.7916/
# /dr_slo_fundus_FAS/dr_slo_fundus_race_fAS/convnext_slo_fundus_lr1e-4_bz10_seed13_auc0.7773/
# /dr_slo_fundus_FAS/dr_slo_fundus_race_fAS/resnet_slo_fundus_lr1e-4_bz10_seed13_auc0.7317/
#/medailab/medailab/clement/FairAdaptiveScaling/scripts/FairVT/fair_distill_grid_search_results/ViT-B_alpha0.3_temp2.0_fis0.3_lr1e-4_bs64_seed13_auc0.8552_fair_distill_MAIN/pred_gt_best_fair_distill_epoch.npz
# /medailab/medailab/clement/FairAdaptiveScaling/scripts/FairVT/fair_distill_grid_search_results/ViT-B_alpha0.3_temp2.0_fis0.3_lr1e-4_bs64_seed13_auc0.8552_fair_distill_MAIN/pred_gt_best_fair_distill_epoch.npz"

# /results_harvard10k_FAS_distill/dr_slo_fundus_race/mobilenet_slo_fundus_lr1e-4_bz10_alpha0.5_temp4.0_seed13_auc0.8186_distill
# /results_harvard10k_FAS_distill/dr_slo_fundus_race/densenet_slo_fundus_lr5e-4_bz64_alpha0.5_temp4.0_seed13_auc0.8245_distill
# /results_harvard10k_FAS_distill/dr_slo_fundus_race/efficientnet_slo_fundus_lr1e-4_bz10_alpha0.5_temp4.0_seed13_auc0.8191_distill/
# /results_harvard10k_FAS_distill/dr_slo_fundus_race/deit_small_slo_fundus_lr1e-4_bz32_alpha0.5_temp4.0_seed13_auc0.7823_distill/
# /results_harvard10k_FAS_distill/dr_slo_fundus_race/swin_t_slo_fundus_lr1e-4_bz32_alpha0.5_temp4.0_seed13_auc0.8325_distill/
# /results_harvard10k_FAS_distill/dr_slo_fundus_race/convnext_slo_fundus_lr1e-4_bz10_alpha0.5_temp4.0_seed13_auc0.8314_distill/
# /results_harvard10k_FAS_distill/dr_slo_fundus_race/resnet_slo_fundus_lr5e-4_bz10_alpha0.5_temp4.0_seed13_auc0.7825_distill/
# /results_harvard10k_FAS_distill/dr_slo_fundus_race/ViT-B_slo_fundus_lr1e-4_bz64_alpha0.5_temp4.0_seed13_auc0.6570_distill/
#r"/medailab/medailab/clement/FairAdaptiveScaling/scripts/FairTeacher/fair_distill_grid_search_results/resnet_alpha0.3_temp2.0_fis0.5_lr1e-4_bs32_seed13_auc0.7649_fair_distill_MAIN/pred_gt_best_fair_distill_epoch.npz"

NPZ_PATH = r"/medailab/medailab/clement/FairAdaptiveScaling/scripts/results_harvard10k_NOFAS_distill/dr_slo_fundus_race/ViT-B_slo_fundus_lr1e-4_bz64_alpha0.5_temp1.0_seed13_auc0.6586_distill/pred_gt_best_distill_epoch.npz"

# Set the target specificity (e.g. 0.95) — the script will find the threshold
# whose specificity is closest to this value and report the resulting sensitivity.
TARGET_SPEC = 0.85 # Adjust the target specificity as needed

def find_threshold_for_specificity(scores, y, target_specificity):
    """Find threshold whose specificity is closest to target_specificity.

    Returns (threshold, specificity_at_threshold, (tp, tn, fp, fn)).
    """
    unique = np.unique(scores)
    if unique.size > 0:
        thresholds = np.concatenate(([unique.max() + 1.0], unique, [unique.min() - 1.0]))
    else:
        thresholds = np.array([0.5])

    best = None
    best_diff = float('inf')
    for t in thresholds:
        preds = (scores >= t).astype(np.int64)
        tp = int(np.sum((preds == 1) & (y == 1)))
        tn = int(np.sum((preds == 0) & (y == 0)))
        fp = int(np.sum((preds == 1) & (y == 0)))
        fn = int(np.sum((preds == 0) & (y == 1)))
        spec = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
        diff = abs(spec - target_specificity)
        if diff < best_diff:
            best_diff = diff
            best = (t, spec, (tp, tn, fp, fn))
    return best


def main():
    data = np.load(NPZ_PATH, allow_pickle=True)
    print('NPZ keys:', list(data.keys()))
    scores = np.asarray(data['test_pred'])
    y = np.asarray(data['test_gt']).astype(np.int64)
    attrs = np.asarray(data['test_attr'])
    print('scores shape:', scores.shape)
    print('y shape:', y.shape)
    print('attrs shape:', attrs.shape)

    # Find the threshold that gives specificity closest to TARGET_SPEC
    thr, spec_overall, counts = find_threshold_for_specificity(scores, y, TARGET_SPEC)
    tp, tn, fp, fn = counts
    sens_overall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    print('\nOverall selected threshold:', thr)
    print(f'Overall counts: TP={tp}, TN={tn}, FP={fp}, FN={fn}')
    print(f'Overall specificity (observed): {spec_overall:.4f}')
    print(f'Overall sensitivity at that specificity: {sens_overall:.4f}\n')

    # mapping provided: race=row0, gender=row1, ethnicity=row2
    mapping = {'race': 0, 'gender': 1, 'ethnicity': 2}
    for name, row in mapping.items():
        arr = attrs[row]
        u, counts = np.unique(arr, return_counts=True)
        print(f"Subgroup '{name}' (row {row}) unique values: {list(u)}")
        for val, cnt in zip(u, counts):
            mask = (arr == val)
            n = int(mask.sum())
            if n == 0:
                print(f"  Value={val}: no samples, skipped")
                continue
            sub_scores = scores[mask]
            sub_y = y[mask]
            preds_sub = (sub_scores >= thr).astype(np.int64)
            tp = int(np.sum((preds_sub == 1) & (sub_y == 1)))
            tn = int(np.sum((preds_sub == 0) & (sub_y == 0)))
            fp = int(np.sum((preds_sub == 1) & (sub_y == 0)))
            fn = int(np.sum((preds_sub == 0) & (sub_y == 1)))
            sens_sub = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
            print(f"  Value={val}: N={n}, Sensitivity={sens_sub:.4f} (TP={tp}, TN={tn}, FP={fp}, FN={fn})")
        print()

if __name__ == '__main__':
    main()

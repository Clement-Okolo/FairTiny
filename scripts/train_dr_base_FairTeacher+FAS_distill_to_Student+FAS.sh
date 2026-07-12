#!/bin/bash
# Grid search for hyperparameter optimization in fair knowledge distillation

DATASET_DIR=/medailab/medailab/shilab/FairVision
RESULT_DIR=.

MODALITY_TYPE='slo_fundus' # Options: 'oct_bscans_3d' | 'slo_fundus'
ATTRIBUTE_TYPE=race # Options: race | gender | hispanic

#TEACHER_MODEL_PATH="/scratch/clement/FairAdaptiveScaling/scripts/results_harvard10k_FAS/ViT-B_slo_fundus_lr1e-4_bz64_seed13_auc0.8532/model_best_epoch.pth"

TEACHER_MODEL_PATH="/medailab/medailab/clement/FairAdaptiveScaling/scripts/results_harvard10k_student+FAS/dr_slo_fundus_FAS/dr_slo_fundus_race_fAS/ViT-B_slo_fundus_lr1e-4_bz64_seed13_auc0.8532/model_best_epoch.pth"

# Hyperparameter search space for distillation and FIS
#DISTILL_ALPHA=( 0.3 0.5 0.7 )
#DISTILL_TEMP=( 1.0 2.0 4.0 )
#SCALE_COEF=( 0.3 0.5 0.7 )
MODEL_TYPE=(deit_small) # Options: deit_small | efficientnet | resnet | swin_t | densenet | swin | convnext | mobilenet | ViT-B
#swin_t densenet convnext mobilenet

NUM_EPOCH=20
BLR=5e-4
DP=0.1
LD=0.55
VIT_WEIGHTS=imagenet

# Create results log directory
GRID_SEARCH_DIR="${RESULT_DIR}/FairTeacher_20Epoch/fair_distill_grid_search_results"
mkdir -p ${GRID_SEARCH_DIR}

# Results log file
RESULTS_CSV="${GRID_SEARCH_DIR}/fair_distill_grid_search_results.csv"
echo "model,alpha,temp,fis_coef,lr,batch_size,weight_decay,val_auc,test_auc,best_epoch,path" > ${RESULTS_CSV}

# Loop through models
for (( m=0; m<${#MODEL_TYPE[@]}; m++ ));
do
    # Set model-specific parameters
    case ${MODEL_TYPE[$m]} in
        "deit_small")
            LR=( 1e-4)
            WD=0.05
            BATCH_SIZE=( 64 )
            DISTILL_ALPHA=( 0.7 )
            DISTILL_TEMP=( 2.0 )
            SCALE_COEF=( 0.7 )
            ;;
        "ViT-B")
            LR=1e-4
            WD=0.01
            BATCH_SIZE=64
            DISTILL_ALPHA=( 0.5 )
            DISTILL_TEMP=( 1.0 )
            SCALE_COEF=( 0.3 )
            ;;
        "efficientnet")
            LR=( 5e-4 )
            WD=0.01
            BATCH_SIZE=( 64 )
            DISTILL_ALPHA=( 0.5 )
            DISTILL_TEMP=( 2.0 )
            SCALE_COEF=( 0.7 )
            ;;
        "resnet")
            LR=( 1e-4 )
            WD=0.01
            BATCH_SIZE=( 32 )
            DISTILL_ALPHA=( 0.3 )
            DISTILL_TEMP=( 2.0 )
            SCALE_COEF=( 0.5 )
            ;;
        "mobilenet")
            LR=( 5e-4 )
            WD=0.01
            BATCH_SIZE=( 32 )
            DISTILL_ALPHA=( 0.5 )
            DISTILL_TEMP=( 1.0 )
            SCALE_COEF=( 0.7 )
            ;;
        "convnext")
            LR=( 1e-4 )
            WD=0.05
            BATCH_SIZE=( 32 )
            DISTILL_ALPHA=( 0.3 )
            DISTILL_TEMP=( 2.0 )
            SCALE_COEF=( 0.5 )
            ;;
        "vgg")
            LR=( 1e-5 5e-5 )
            WD=0.0           
            BATCH_SIZE=( 10 32 )
            ;;
        "swin")
            LR=( 1e-4 5e-4 )
            WD=0.0
            BATCH_SIZE=( 32 64 )
            ;;
        "densenet")
            LR=( 5e-4 )
            WD=0.0 
            BATCH_SIZE=( 32)
            DISTILL_ALPHA=( 0.7 )
            DISTILL_TEMP=( 4.0 )
            SCALE_COEF=( 0.3 )
            ;;
        "swin_t")
            LR=( 1e-4 )
            WD=0.05
            BATCH_SIZE=( 32 )
            DISTILL_ALPHA=( 0.7 )
            DISTILL_TEMP=( 2.0 )
            SCALE_COEF=( 0.5 )
            ;;
        *)
            LR=( 1e-4 )
            WD=0.05
            BATCH_SIZE=( 32 )
            ;;
    esac

    # Loop through hyperparameters for both distillation and FIS
    for (( a=0; a<${#DISTILL_ALPHA[@]}; a++ ));
    do
        for (( t=0; t<${#DISTILL_TEMP[@]}; t++ ));
        do
            for (( f=0; f<${#SCALE_COEF[@]}; f++ ));
            do
                for (( l=0; l<${#LR[@]}; l++ ));
                do
                    for (( b=0; b<${#BATCH_SIZE[@]}; b++ ));
                    do
                        # Create run-specific directory
                        RUN_DIR="${GRID_SEARCH_DIR}/${MODEL_TYPE[$m]}_alpha${DISTILL_ALPHA[$a]}_temp${DISTILL_TEMP[$t]}_fis${SCALE_COEF[$f]}_lr${LR[$l]}_bs${BATCH_SIZE[$b]}"
                        mkdir -p ${RUN_DIR}
                        
                        echo "Running ${MODEL_TYPE[$m]} with alpha=${DISTILL_ALPHA[$a]}, temp=${DISTILL_TEMP[$t]}, fis_coef=${SCALE_COEF[$f]}, lr=${LR[$l]}, bs=${BATCH_SIZE[$b]}"
                        
                        # FIS group weights - keeping them equal for all groups
                        SCALE_GROUP_WEIGHT=1.0
                        
                        # Run training with these hyperparameters
                        python train_dr_fair_distill_fis.py \
                            --data_dir ${DATASET_DIR}/DR/ \
                            --result_dir ${RUN_DIR} \
                            --model_type ${MODEL_TYPE[$m]} \
                            --image_size 224 \
                            --lr ${LR[$l]} \
                            --weight-decay ${WD} \
                            --momentum 0.1 \
                            --batch_size ${BATCH_SIZE[$b]} \
                            --epochs ${NUM_EPOCH} \
                            --modality_types ${MODALITY_TYPE} \
                            --perf_file "grid_search_${MODEL_TYPE[$m]}_fair_distill.csv" \
                            --attribute_type ${ATTRIBUTE_TYPE} \
                            --vit_weights ${VIT_WEIGHTS} \
                            --blr ${BLR} \
                            --drop_path ${DP} \
                            --layer_decay ${LD} \
                            --weight_decay ${WD} \
                            --need_balance False \
                            --teacher_model_type 'ViT-B' \
                            --teacher_model_path ${TEACHER_MODEL_PATH} \
                            --distill_alpha ${DISTILL_ALPHA[$a]} \
                            --distill_temp ${DISTILL_TEMP[$t]} \
                            --fair_scaling_coef ${SCALE_COEF[$f]} \
                            --fair_scaling_group_weights ${SCALE_GROUP_WEIGHT} ${SCALE_GROUP_WEIGHT} ${SCALE_GROUP_WEIGHT} \
                            2>&1 | tee ${RUN_DIR}/training.log
                        
                        # Extract results
                        BEST_EPOCH=$(grep "best AUC" ${RUN_DIR}/training.log | tail -1 | awk '{print $6}')
                        VAL_AUC=$(grep "best val AUC" ${RUN_DIR}/training.log | tail -1 | awk '{print $5}')
                        TEST_AUC=$(grep "best AUC" ${RUN_DIR}/training.log | tail -1 | awk '{print $4}')
                        
                        # Log results
                        echo "${MODEL_TYPE[$m]},${DISTILL_ALPHA[$a]},${DISTILL_TEMP[$t]},${SCALE_COEF[$f]},${LR[$l]},${BATCH_SIZE[$b]},${WD},${VAL_AUC},${TEST_AUC},${BEST_EPOCH},${RUN_DIR}" >> ${RESULTS_CSV}
                        
                        echo "Completed run. Val AUC: ${VAL_AUC}, Test AUC: ${TEST_AUC}, Best Epoch: ${BEST_EPOCH}"
                        echo "---------------------------------------------------------"
                    done
                done
            done
        done
    done
done

# Find best configurations
echo ""
echo "===== Fair Distillation Grid Search Results ====="
echo "Top 5 configurations by validation AUC:"
sort -t, -k8,8nr ${RESULTS_CSV} | head -6

echo ""
echo "Top 5 configurations by test AUC:"
sort -t, -k9,9nr ${RESULTS_CSV} | head -6

echo ""
echo "======= Group Fairness Analysis ======="
echo "Top 5 models by demographic parity (sorted by test_dpd):"
# Extract and sort results by demographic parity difference (lower is better)
grep -o 'test_dpd_attr0 [0-9.]*' ${GRID_SEARCH_DIR}/*/training.log | sort -k2,2n | head -5

echo ""
echo "Top 5 models by equalized odds (sorted by test_eod):"
# Extract and sort results by equalized odds difference (lower is better)
grep -o 'test_eod_attr0 [0-9.]*' ${GRID_SEARCH_DIR}/*/training.log | sort -k2,2n | head -5
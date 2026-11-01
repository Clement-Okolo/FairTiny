#!/bin/bash
# 3D baseline knowledge distillation (no fairness term) on OCT volumes.
# Teacher: a 3D model trained with train_dr_fair_real3d.py (baseline).
# Each GPU in GPUS gets its own queue; models are dealt out round-robin
# (with GPUS=( 0 1 4 ): model 1 -> GPU 0, model 2 -> GPU 1, model 3 -> GPU 4, model 4 -> GPU 0, ...).

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

mkdir -p /medailab/medailab/clement/tmp
export TMPDIR=/medailab/medailab/clement/tmp

DATASET_DIR=/medailab/medailab/shilab/FairVision
RESULT_DIR=.
LOG_DIR=${RESULT_DIR}/logs_real3d_baseline_distill
mkdir -p ${LOG_DIR}

# ---- teacher (EDIT THESE) ---------------------------------------------------------------
TEACHER_MODEL_TYPE=resnet   # mobilenet | efficientnet | densenet | resnet | vgg
TEACHER_MODEL_PATH="/path/to/results_OCT_3D/.../dr_oct_bscans_3d_race_resnet_Conv3d_..._seed13_aucX.XXXX/model_best_epoch.pth"
# -----------------------------------------------------------------------------------------

MODEL_TYPE=( mobilenet efficientnet densenet resnet vgg )   # students
GPUS=( 0 1 4 )

MODALITY_TYPE='oct_bscans_3d'
ATTRIBUTE_TYPE=race   # race | gender | hispanic
CONV_TYPE=Conv3d
NUM_EPOCH=20
NEED_BALANCE=false

DISTILL_ALPHA=0.5
DISTILL_TEMP=4.0

if [ ! -f "${TEACHER_MODEL_PATH}" ]; then
    echo "Teacher checkpoint not found: ${TEACHER_MODEL_PATH}"
    echo "Set TEACHER_MODEL_PATH at the top of this script."
    exit 1
fi

for (( g=0; g<${#GPUS[@]}; g++ ));
do
(
for (( m=g; m<${#MODEL_TYPE[@]}; m+=${#GPUS[@]} ));
do
    case ${MODEL_TYPE[$m]} in
        "mobilenet")    LR=( 1e-4 ); WD=0.01; BATCH_SIZE=( 2 ) ;;
        "efficientnet") LR=( 1e-4 ); WD=0.01; BATCH_SIZE=( 2 ) ;;
        "densenet")     LR=( 5e-4 ); WD=0.0;  BATCH_SIZE=( 2 ) ;;
        "resnet")       LR=( 1e-4 ); WD=0.01; BATCH_SIZE=( 2 ) ;;
        "vgg")          LR=( 1e-5 ); WD=0.0;  BATCH_SIZE=( 2 ) ;;
        *)              LR=( 1e-4 ); WD=0.05; BATCH_SIZE=( 2 ) ;;
    esac

    for (( l=0; l<${#LR[@]}; l++ ));
    do
    for (( b=0; b<${#BATCH_SIZE[@]}; b++ ));
    do
        RUN_NAME=dr_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}_${MODEL_TYPE[$m]}_from_${TEACHER_MODEL_TYPE}_${CONV_TYPE}_distill_alpha${DISTILL_ALPHA}_temp${DISTILL_TEMP}_lr${LR[$l]}_bz${BATCH_SIZE[$b]}
        PERF_FILE=${MODEL_TYPE[$m]}_${MODALITY_TYPE}_${CONV_TYPE}_${ATTRIBUTE_TYPE}_3D_distill.csv
        LOG_FILE=${LOG_DIR}/${RUN_NAME}.txt

        echo "[GPU ${GPUS[$g]}] start ${MODEL_TYPE[$m]} lr${LR[$l]} -> ${LOG_FILE}"

        CUDA_VISIBLE_DEVICES=${GPUS[$g]} python ${SCRIPT_DIR}/train_real3d_baseline_distillation.py \
            --data_dir ${DATASET_DIR}/DR/ \
            --result_dir ${RESULT_DIR}/results_OCT_3D/Distillation/${RUN_NAME} \
            --model_type ${MODEL_TYPE[$m]} \
            --image_size 224 \
            --lr ${LR[$l]} --weight-decay ${WD} --momentum 0.1 \
            --batch_size ${BATCH_SIZE[$b]} \
            --epochs ${NUM_EPOCH} \
            --modality_types ${MODALITY_TYPE} \
            --perf_file ${PERF_FILE} \
            --attribute_type ${ATTRIBUTE_TYPE} \
            --conv_type ${CONV_TYPE} \
            --need_balance ${NEED_BALANCE} \
            --teacher_model_type ${TEACHER_MODEL_TYPE} \
            --teacher_model_path ${TEACHER_MODEL_PATH} \
            --distill_alpha ${DISTILL_ALPHA} \
            --distill_temp ${DISTILL_TEMP} > ${LOG_FILE} 2>&1

        echo "[GPU ${GPUS[$g]}] done  ${MODEL_TYPE[$m]} lr${LR[$l]} (exit code $?)"
    done    # end of batch-size loop
    done    # end of learning-rate loop
done        # end of model loop
) &
done        # end of GPU loop

wait
echo "All runs finished."

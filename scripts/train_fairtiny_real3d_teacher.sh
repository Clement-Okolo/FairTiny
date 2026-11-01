#!/bin/bash
# FairTiny TEACHER configuration on 3D OCT volumes:
#   knowledge distillation + Fair Instance Scaling, distilled from a FAIR 3D teacher.
# Teacher: a 3D model trained with train_dr_fair_real3d_fis.py (FAS), i.e. step (a) of the
#   FairTiny teacher recipe; this script is step (b).
# Each GPU in GPUS gets its own queue; models are dealt out round-robin
# (with GPUS=( 0 1 4 ): model 1 -> GPU 0, model 2 -> GPU 1, model 3 -> GPU 4, model 4 -> GPU 0, ...).

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

mkdir -p /medailab/medailab/clement/tmp
export TMPDIR=/medailab/medailab/clement/tmp

DATASET_DIR=/medailab/medailab/shilab/FairVision
RESULT_DIR=.
EXPERIMENT=FairTiny_Teacher
LOG_DIR=${RESULT_DIR}/logs_real3d_${EXPERIMENT}
mkdir -p ${LOG_DIR}

# ---- teacher (EDIT THESE) ---------------------------------------------------------------
# fair (FAS) teacher, from train_dr_fair_real3d_fis.py
TEACHER_MODEL_TYPE=resnet   # mobilenet | efficientnet | densenet | resnet | vgg
TEACHER_MODEL_PATH="/path/to/results_OCT_3D/FAS/dr_oct_bscans_3d_race_resnet_Conv3d_FAS_..._seed13_aucX.XXXX/model_best_epoch.pth"
# -----------------------------------------------------------------------------------------

MODEL_TYPE=( mobilenet efficientnet densenet resnet vgg )   # students
GPUS=( 0 1 4 )

MODALITY_TYPE='oct_bscans_3d'
ATTRIBUTE_TYPE=race   # race | gender | hispanic
CONV_TYPE=Conv3d
NUM_EPOCH=20
NEED_BALANCE=false

# FIS group weights - keeping them equal for all groups (as in the 2D FairTiny scripts)
SCALE_GROUP_WEIGHT=1.0

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
    # LR / WD / batch size: same as the 3D baselines.
    # alpha / temp / FIS coef: best values from the 2D FairTiny grid search (vgg was not tuned in 2D).
    # Put several values in any ( ) list to grid-search it.
    case ${MODEL_TYPE[$m]} in
        "mobilenet")    LR=( 1e-4 ); WD=0.01; BATCH_SIZE=( 2 ); DISTILL_ALPHA=( 0.5 ); DISTILL_TEMP=( 1.0 ); SCALE_COEF=( 0.7 ) ;;
        "efficientnet") LR=( 1e-4 ); WD=0.01; BATCH_SIZE=( 2 ); DISTILL_ALPHA=( 0.5 ); DISTILL_TEMP=( 2.0 ); SCALE_COEF=( 0.7 ) ;;
        "densenet")     LR=( 5e-4 ); WD=0.0;  BATCH_SIZE=( 2 ); DISTILL_ALPHA=( 0.7 ); DISTILL_TEMP=( 4.0 ); SCALE_COEF=( 0.3 ) ;;
        "resnet")       LR=( 1e-4 ); WD=0.01; BATCH_SIZE=( 2 ); DISTILL_ALPHA=( 0.3 ); DISTILL_TEMP=( 2.0 ); SCALE_COEF=( 0.5 ) ;;
        "vgg")          LR=( 1e-5 ); WD=0.0;  BATCH_SIZE=( 2 ); DISTILL_ALPHA=( 0.5 ); DISTILL_TEMP=( 2.0 ); SCALE_COEF=( 0.5 ) ;;
        *)              LR=( 1e-4 ); WD=0.05; BATCH_SIZE=( 2 ); DISTILL_ALPHA=( 0.5 ); DISTILL_TEMP=( 2.0 ); SCALE_COEF=( 0.5 ) ;;
    esac

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
        RUN_NAME=dr_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}_${MODEL_TYPE[$m]}_from_${TEACHER_MODEL_TYPE}_${CONV_TYPE}_alpha${DISTILL_ALPHA[$a]}_temp${DISTILL_TEMP[$t]}_fis${SCALE_COEF[$f]}_lr${LR[$l]}_bz${BATCH_SIZE[$b]}
        PERF_FILE=${MODEL_TYPE[$m]}_${MODALITY_TYPE}_${CONV_TYPE}_${ATTRIBUTE_TYPE}_3D_fairtiny_teacher.csv
        LOG_FILE=${LOG_DIR}/${RUN_NAME}.txt

        echo "[GPU ${GPUS[$g]}] start ${MODEL_TYPE[$m]} alpha${DISTILL_ALPHA[$a]} temp${DISTILL_TEMP[$t]} fis${SCALE_COEF[$f]} lr${LR[$l]} -> ${LOG_FILE}"

        CUDA_VISIBLE_DEVICES=${GPUS[$g]} python ${SCRIPT_DIR}/train_real3d_distill_do.py \
            --data_dir ${DATASET_DIR}/DR/ \
            --result_dir ${RESULT_DIR}/results_OCT_3D/${EXPERIMENT}/${RUN_NAME} \
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
            --distill_alpha ${DISTILL_ALPHA[$a]} \
            --distill_temp ${DISTILL_TEMP[$t]} \
            --fair_scaling_coef ${SCALE_COEF[$f]} \
            --fair_scaling_group_weights ${SCALE_GROUP_WEIGHT} ${SCALE_GROUP_WEIGHT} ${SCALE_GROUP_WEIGHT} > ${LOG_FILE} 2>&1

        echo "[GPU ${GPUS[$g]}] done  ${MODEL_TYPE[$m]} (exit code $?)"
    done    # end of batch-size loop
    done    # end of learning-rate loop
    done    # end of FIS-coef loop
    done    # end of temperature loop
    done    # end of alpha loop
done        # end of model loop
) &
done        # end of GPU loop

wait
echo "All runs finished."

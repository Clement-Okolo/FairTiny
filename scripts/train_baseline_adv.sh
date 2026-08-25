#!/bin/bash

DATASET_DIR=/your/path/FairVision
RESULT_DIR=.

MODEL_TYPE=( efficientnet ) # Options: convnext | mobilevit | deit_small | swin_t | mobilenet | efficientnet | vit | resnet | swin | vgg | resnext | wideresnet | efficientnetv1
NUM_EPOCH=20
MODALITY_TYPE='slo_fundus' # Options: 'oct_bscans_3d' | 'slo_fundus'
ATTRIBUTE_TYPE=race # Options: race | gender | hispanic

# OPTIMIZER='adamw'
# OPTIMIZER_ARGUMENTS='{"lr": 0.001, "weight_decay": 0.01}'

# SCHEDULER='step_lr'
# SCHEDULER_ARGUMENTS='{"step_size": 30, "gamma": 0.1}'

if [ ${MODALITY_TYPE} = 'oct_bscans' ]; then
	LR=1e-4
	BATCH_SIZE=6
elif [ ${MODALITY_TYPE} = 'slo_fundus' ]; then
	LR=1e-4
# 	LR=( 1e-3 1e-4 1e-5 1e-6 1e-7 )
	BATCH_SIZE=64
else
	LR=1e-4
	BATCH_SIZE=6
fi

VIT_WEIGHTS=imagenet
BLR=5e-4
WD=0.
LD=0.55
DP=0.1

# NEED_BALANCED=True 

MODEL_TYPE=( convnext )
# MODEL_TYPE=( mobilenet convnext deit_small swin_t efficientnet densenet resnet swin convnext ViT-B vgg )
ATTRIBUTE_TYPE=( race ) # ( gender ethnicity maritalstatus language gender_race gender_ethnicity )

for (( j=0; j<${#MODEL_TYPE[@]}; j++ ));
do
for (( a=0; a<${#ATTRIBUTE_TYPE[@]}; a++ ));
do

NUM_EPOCH=10
BATCH_SIZE=10
if [ ${MODEL_TYPE[$j]} = 'ViT-B' ]; then
    # NEED_BALANCED=True
    WD=0.01
    NUM_EPOCH=20
    BATCH_SIZE=64
fi

if [ ${MODEL_TYPE[$j]} = 'vgg' ]; then
    # NEED_BALANCED=True
    LR=1e-5
    WD=0.
    BATCH_SIZE=64
    NUM_EPOCH=20
fi

if [ ${MODEL_TYPE[$j]} = 'efficientnet' ]; then
    # NEED_BALANCED=True
    LR=1e-4
    WD=0.0
    BATCH_SIZE=32
fi

if [ ${MODEL_TYPE[$j]} = 'swin' ]; then
    # NEED_BALANCED=True
    LR=1e-4
    WD=0.
    BATCH_SIZE=64
    NUM_EPOCH=10
fi

if [ ${MODEL_TYPE[$j]} = 'densenet' ]; then
    # NEED_BALANCED=True
    LR=5e-4
    WD=0.
    BATCH_SIZE=64
    NUM_EPOCH=10
fi

if [ ${MODEL_TYPE[$j]} = 'swin_t' ]; then
    NEED_BALANCED=True
    LR=1e-4
    WD=0.05
    BATCH_SIZE=32
    NUM_EPOCH=10
fi

if [ ${MODEL_TYPE[$j]} = 'deit_small' ]; then
    NEED_BALANCED=True
    LR=1e-4
    WD=0.05
    BATCH_SIZE=32
    NUM_EPOCH=10
fi

if [ ${MODEL_TYPE[$j]} = 'levit' ]; then
    NEED_BALANCED=True
    LR=1e-4
    WD=0.01
    BATCH_SIZE=64
    NUM_EPOCH=10
fi

PERF_FILE=${MODEL_TYPE[$j]}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE[$a]}_adv.csv

python train_baseline_adv.py \
		--data_dir ${DATASET_DIR}/DR/ \
		--result_dir ${RESULT_DIR}/results_harvard10k/dr_slo_fundus_adversarial/dr_${MODALITY_TYPE}_${ATTRIBUTE_TYPE[$a]}/${MODEL_TYPE[$j]}_${MODALITY_TYPE}_lr${LR}_bz${BATCH_SIZE}_adv \
		--model_type ${MODEL_TYPE[$j]} \
		--image_size 224 \
		--lr ${LR} --weight-decay 0. --momentum 0.1 \
		--batch_size ${BATCH_SIZE} \
		--epochs ${NUM_EPOCH} \
		--modality_types ${MODALITY_TYPE} \
		--perf_file ${PERF_FILE} \
        --attribute_type ${ATTRIBUTE_TYPE[$a]} \
        --vit_weights ${VIT_WEIGHTS} \
        --blr ${BLR} \
        --drop_path ${DP} \
        --layer_decay ${LD} \
        --weight_decay ${WD} \
#         --need_balance ${NEED_BALANCED}
		# --optimizer ${OPTIMIZER} \
		# --optimizer_arguments ${OPTIMIZER_ARGUMENTS} \
		# --scheduler ${SCHEDULER} \
		# --scheduler_arguments ${SCHEDULER_ARGUMENTS}
done
done

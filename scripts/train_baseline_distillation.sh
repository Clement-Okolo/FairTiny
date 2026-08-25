#!/bin/bash
# Knowledge distillation for medical imaging

DATASET_DIR=/your/path/FairVision
RESULT_DIR=.

MODALITY_TYPE='slo_fundus' 
ATTRIBUTE_TYPE=race # Options: race | gender | hispanic

# Path to teacher model
#TEACHER_MODEL_TYPE= 'ViT-B'

TEACHER_MODEL_PATH="/your/path/model_best_epoch.pth"

# Knowledge distillation parameters
DISTILL_ALPHA=0.5
DISTILL_TEMP=4.0 #2.0
if [ ${MODALITY_TYPE} = 'oct_bscans' ]; then
	LR=1e-4 
	BATCH_SIZE=6
elif [ ${MODALITY_TYPE} = 'slo_fundus' ]; then
	LR=1e-4
	BATCH_SIZE=10
else
	LR=1e-4
	BATCH_SIZE=6
fi

BLR=5e-4
WD=6e-5 # 0.0001
LD=0.55
DP=0.1
VIT_WEIGHTS=imagenet

NEED_BALANCED=False

# Select Student Model
MODEL_TYPE=( ViT-B ) # Options - levit mobilevit deit_small swin_t efficientnet densenet resnet swin convnext ViT-B mobilenet

ATTRIBUTE_TYPE=( race )

for (( j=0; j<${#MODEL_TYPE[@]}; j++ ));
do
for (( a=0; a<${#ATTRIBUTE_TYPE[@]}; a++ ));
do

NUM_EPOCH=20
BATCH_SIZE=10
PERF_FILE=${MODEL_TYPE[$j]}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE[$a]}_distill.csv

# Set model-specific parameters (following your original script)
if [ ${MODEL_TYPE[$j]} = 'efficientnet' ]; then
    LR=1e-4
    WD=0.01
    BATCH_SIZE=10
fi

if [ ${MODEL_TYPE[$j]} = 'ViT-B' ]; then
    LR=1e-4
    WD=0.01
    BATCH_SIZE=64
fi

if [ ${MODEL_TYPE[$j]} = 'mobilenet' ]; then
    LR=1e-4
    WD=0.01
    BATCH_SIZE=10
fi

if [ ${MODEL_TYPE[$j]} = 'swin' ]; then
    LR=1e-4
    WD=0.
    BATCH_SIZE=64
    NUM_EPOCH=10
fi

if [ ${MODEL_TYPE[$j]} = 'densenet' ]; then
    LR=5e-4
    WD=0.
    BATCH_SIZE=64
    NUM_EPOCH=10
fi

if [ ${MODEL_TYPE[$j]} = 'swin_t' ]; then
    LR=1e-4
    WD=0.05
    BATCH_SIZE=32
    NUM_EPOCH=10
fi

if [ ${MODEL_TYPE[$j]} = 'deit_small' ]; then
    LR=1e-4
    WD=0.05
    BATCH_SIZE=32
    NUM_EPOCH=10
fi

if [ ${MODEL_TYPE[$j]} = 'levit' ]; then
    LR=1e-4
    WD=0.01
    BATCH_SIZE=64
    NUM_EPOCH=10
fi

if [ ${MODEL_TYPE[$j]} = 'mobilevit' ]; then
    LR=1e-4
    WD=0.01
    BATCH_SIZE=32
    NUM_EPOCH=10
fi

# Run the distillation training
python train_baseline_distillation.py \
		--data_dir ${DATASET_DIR}/DR/ \
		--result_dir ${RESULT_DIR}/results_harvard10k_FAS_distill/dr_${MODALITY_TYPE}_${ATTRIBUTE_TYPE[$a]}/${MODEL_TYPE[$j]}_${MODALITY_TYPE}_lr${LR}_bz${BATCH_SIZE}_alpha${DISTILL_ALPHA}_temp${DISTILL_TEMP} \
		--model_type ${MODEL_TYPE[$j]} \
		--image_size 224 \
		--lr ${LR} --weight-decay ${WD} --momentum 0.1 \
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
        --need_balance False \
        --teacher_model_type 'ViT-B' \
        --teacher_model_path ${TEACHER_MODEL_PATH} \
        --distill_alpha ${DISTILL_ALPHA} \
        --distill_temp ${DISTILL_TEMP}

done
done

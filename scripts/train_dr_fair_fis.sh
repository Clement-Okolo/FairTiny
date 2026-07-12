# #!/bin/bash
DATASET_DIR=/medailab/medailab/shilab/FairVision
# DATASET_DIR=/data/home/shim/pyspace/others/pyspace/ICLR_30k/ODIR_Data/ODIR_Data_New_seed10
RESULT_DIR=.
MODEL_TYPE=( vit ) # Options: convnext | mobilevit | deit_small | swin_t | mobilenet | efficientnet | vit | resnet | swin | vgg | resnext | wideresnet | efficientnetv1
NUM_EPOCH=10
MODALITY_TYPE='slo_fundus' # Options: 'oct_bscans_3d' | 'slo_fundus'
ATTRIBUTE_TYPE=race # Options: race | gender | hispanic

if [ ${MODALITY_TYPE} = 'oct_bscans_3d' ]; then
	LR=1e-4 
	BATCH_SIZE=6
elif [ ${MODALITY_TYPE} = 'slo_fundus' ]; then
	LR=1e-4
	BATCH_SIZE=10
else
	LR=1e-4
	BATCH_SIZE=6
fi

SCALE_COEF=0.7
SCALE_GROUP_WEIGHT=1.
LOSS_TYPE='bce'

BLR=5e-4
WD=6e-5 # 0.0001
LD=0.55
DP=0.1
VIT_WEIGHTS=imagenet


# ATTRIBUTE_TYPE=race # Options: race | gender | hispanic
# MODEL_TYPE=ViT-B
# VIT_WEIGHTS=imagenet # [scratch, mae, mocov3, mae_chest_xray, mae_color_fundus, dinov2, dinov2_registers,imagenet]
# BATCH_SIZE=64 # [64, 128, 256]
# BLR=5e-4 # [1e-4, 5e-4, 1e-3]
# WD=0.01 # [0.01, 0.05, 0.1]
# LD=0.55 # [0.55, 0.65, 0.75]
# DP=0.1 # [0.05, 0.1, 0.2]
# EXP_NAME=${VIT_WEIGHTS}_S13

# PERF_FILE=DR_${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME}.csv
# python scripts/train_dr_fair_fis.py --seed 13 --epochs 30 --batch_size ${BATCH_SIZE} --blr ${BLR} --min_lr 1e-6 --warmup_epochs 5 --weight_decay ${WD} --layer_decay ${LD} --drop_path ${DP} --data_dir ${DATASET_DIR}/DR/ --result_dir ${RESULT_DIR}/results_fis_harvard10k/dr_${MODALITY_TYPE}_${ATTRIBUTE_TYPE[$j]}_fis/${MODEL_TYPE}_${MODALITY_TYPE}_lr${LR}_bz${BATCH_SIZE} --model_type ${MODEL_TYPE} --modality_types ${MODALITY_TYPE} --perf_file ${PERF_FILE} --vit_weights ${VIT_WEIGHTS} --fair_scaling_coef ${SCALE_COEF} --fair_scaling_group_weights ${SCALE_GROUP_WEIGHT} ${SCALE_GROUP_WEIGHT} ${SCALE_GROUP_WEIGHT} ${SCALE_GROUP_WEIGHT} ${SCALE_GROUP_WEIGHT} 1.
# # python scripts/train_dr_fair.py --seed 42 --epochs 100 --batch_size ${BATCH_SIZE} --blr ${BLR} --min_lr 1e-6 --warmup_epochs 5 --weight_decay ${WD} --layer_decay ${LD} --drop_path ${DP} --data_dir ${DATASET_DIR}/DR/ --result_dir ${RESULT_DIR}/DR_${MODEL_TYPE}_${MODALITY_TYPE}_${EXP_NAME} --model_type ${MODEL_TYPE} --modality_types ${MODALITY_TYPE} --perf_file ${PERF_FILE} --vit_weights ${VIT_WEIGHTS}


ATTRIBUTE_TYPE=( race ) # ( race gender hispanic maritalstatus language genderrace genderethnicity )

for (( j=0; j<${#ATTRIBUTE_TYPE[@]}; j++ ));
do

if [ ${MODEL_TYPE} = 'ViT-B' ]; then
    # NEED_BALANCED=True
    if [ ${MODALITY_TYPE} = 'slo_fundus' ]; then
        BATCH_SIZE=64
    fi
    WD=0.01
    NUM_EPOCH=10
    LR=1e-4
    BLR=5e-4
fi

if [ ${MODEL_TYPE[$j]} = 'efficientnet' ]; then
    # NEED_BALANCED=True
    LR=1e-4
    WD=0.0
#     BATCH_SIZE=32
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

PERF_FILE=${MODEL_TYPE}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE[$j]}.csv

python train_dr_fair_fis.py \
		--data_dir ${DATASET_DIR}/DR/ \
		--result_dir ${RESULT_DIR}/results_harvard10k_FAS/dr_slo_fundus_FAS/dr_${MODALITY_TYPE}_${ATTRIBUTE_TYPE[$j]}_fAS/${MODEL_TYPE}_${MODALITY_TYPE}_lr${LR}_bz${BATCH_SIZE} \
		--model_type ${MODEL_TYPE} \
		--image_size 224 \
		--loss_type ${LOSS_TYPE} \
		--lr ${LR} --weight-decay 0. --momentum 0.1 \
		--batch_size ${BATCH_SIZE} \
		--epochs ${NUM_EPOCH} \
		--modality_types ${MODALITY_TYPE} \
		--perf_file ${PERF_FILE} \
		--attribute_type ${ATTRIBUTE_TYPE[$j]} \
		--fair_scaling_coef ${SCALE_COEF} \
        --vit_weights ${VIT_WEIGHTS} \
        --blr ${BLR} \
        --drop_path ${DP} \
        --layer_decay ${LD} \
        --weight_decay ${WD} \
		--fair_scaling_group_weights ${SCALE_GROUP_WEIGHT} ${SCALE_GROUP_WEIGHT} ${SCALE_GROUP_WEIGHT} ${SCALE_GROUP_WEIGHT} ${SCALE_GROUP_WEIGHT} 1. 

done



# BLR=5e-4
# WD=6e-5 # 0.0001
# LD=0.55
# DP=0.1
# VIT_WEIGHTS=imagenet

# NEED_BALANCED=False
# # VIT_WEIGHT=dinov2
# MODEL_TYPE=( efficientnet ) # ( efficientnet densenet resnet swin convnext ViT-B vgg )
# ATTRIBUTE_TYPE=( race )

# for (( j=0; j<${#MODEL_TYPE[@]}; j++ ));
# do
# for (( a=0; a<${#ATTRIBUTE_TYPE[@]}; a++ ));
# do

# NUM_EPOCH=10
# BATCH_SIZE=10
# PERF_FILE=${MODEL_TYPE[$j]}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE[$a]}_oversample.csv

# if [ ${MODEL_TYPE[$j]} = 'ViT-B' ]; then
#     # NEED_BALANCED=True
#     if [ ${MODALITY_TYPE} = 'slo_fundus' ]; then
#         BATCH_SIZE=64
#     fi
#     WD=0.01
#     NUM_EPOCH=20
#     LR=1e-4
#     BLR=5e-4
# fi

# if [ ${MODEL_TYPE[$j]} = 'vgg' ]; then
#     # NEED_BALANCED=True
#     LR=1e-5
#     WD=0.0
#     BATCH_SIZE=10
#     NUM_EPOCH=20
# fi

# if [ ${MODEL_TYPE[$j]} = 'efficientnet' ]; then
#     # NEED_BALANCED=True
#     LR=1e-4
#     WD=0.0
#     BATCH_SIZE=10
# fi

# if [ ${MODEL_TYPE[$j]} = 'swin' ]; then
#     # NEED_BALANCED=True
#     LR=1e-4
#     WD=0.
#     BATCH_SIZE=64
#     NUM_EPOCH=10
# fi

# if [ ${MODEL_TYPE[$j]} = 'densenet' ]; then
#     # NEED_BALANCED=True
#     LR=5e-4
#     WD=0.
#     BATCH_SIZE=64
#     NUM_EPOCH=10
# fi

# python ./scripts/train_dr_fair_fis.py \
# 		--data_dir ${DATASET_DIR}/DR/ \
# 		--result_dir ${RESULT_DIR}/results_fis/dr_${MODALITY_TYPE}_${ATTRIBUTE_TYPE[$a]}/${MODEL_TYPE[$j]}_${MODALITY_TYPE}_lr${LR}_bz${BATCH_SIZE} \
# 		--model_type ${MODEL_TYPE[$j]} \
# 		--image_size 200 \
# 		--lr ${LR} --weight-decay 0. --momentum 0.1 \
# 		--batch_size ${BATCH_SIZE} \
# 		--epochs ${NUM_EPOCH} \
# 		--modality_types ${MODALITY_TYPE} \
# 		--perf_file ${PERF_FILE} \
#         --attribute_type ${ATTRIBUTE_TYPE[$a]} \
#         --vit_weights ${VIT_WEIGHTS} \
#         --blr ${BLR} \
#         --drop_path ${DP} \
#         --layer_decay ${LD} \
#         --weight_decay ${WD} \
#         --fair_scaling_coef ${SCALE_COEF} \
#         --fair_scaling_group_weights ${SCALE_GROUP_WEIGHT} ${SCALE_GROUP_WEIGHT} 1. \
#         --need_balance True
# 		# --optimizer ${OPTIMIZER} \
# 		# --optimizer_arguments ${OPTIMIZER_ARGUMENTS} \
# 		# --scheduler ${SCHEDULER} \
# 		# --scheduler_arguments ${SCHEDULER_ARGUMENTS}
# done
# done
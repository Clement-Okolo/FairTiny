# FairTiny
This code is for the paper **FairTiny: Toward Fairer Lightweight Models for Diabetic Retinopathy Prediction**. If you have any questions, please feel free email <okoloclementtochukwu@gmail.com>.


# Requirements

Install the prerequisites:

```
pip install - r requirements.txt
```

# Experiments

1. Run the experiments with the baseline models:
```
./scripts/train_baseline.sh
```

2. Run the experiments with the baseline models with distillation:
```
./scripts/train_baseline_distillation.sh
```

3. Run the experiments with the baseline models with the proposed fairtiny student configuration:
```
./scripts/train_fairtiny_student.sh
```

4. Run the experiments with the baseline models with the proposed fairtiny teacher configuration:

    (a.) Train a fair ViT-B teacher:
    ```
    ./scripts/train_baseline_fas.sh
    ```

    (b.) Execute the fairtiny teacher step:
    ```
    ./scripts/train_fairtiny_teacher.sh
    ```

# 3D OCT experiments (`oct_bscans_3d`)

The 3D scripts distill between 2D ImageNet CNNs converted to 3D (`Conv3d`, i3d weights):
`mobilenet | efficientnet | densenet | resnet | vgg`. Teachers are 3D checkpoints
(`model_best_epoch.pth`) from `train_dr_fair_real3d.py` (baseline) or `train_dr_fair_real3d_fis.py` (FAS).
Each `.sh` runs one queue per GPU in `GPUS=( 0 1 4 )`; set `TEACHER_MODEL_TYPE` and `TEACHER_MODEL_PATH` at the top first.

1. Baseline distillation (baseline 3D teacher):
```
./scripts/train_real3d_baseline_distillation.sh
```

2. FairTiny student configuration (baseline 3D teacher + FIS):
```
./scripts/train_fairtiny_real3d_student.sh
```

3. FairTiny teacher configuration (fair FAS 3D teacher + FIS):
```
./scripts/train_fairtiny_real3d_teacher.sh
```

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

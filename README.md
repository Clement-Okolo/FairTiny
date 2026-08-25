# FairTiny

# Requirements

To install the prerequisites, run:

```
pip install - r requirements.txt
```

# Experiments

1. To run the experiments with the baseline models, execute:
```
./scripts/train_baseline.sh
```

2. To run the experiments with the baseline models with distillation, execute:
```
./scripts/train_baseline_distillation.sh
```

3. To run the experiments with the baseline models with the proposed fairtiny student configuration, execute:
```
./scripts/train_fairtiny_student.sh
```

4. To run the experiments with the baseline models with the proposed fairtiny teacher configuration, execute:

(a.) First train a fair ViT-B teacher:
```
./scripts/train_baseline_fas.sh
```

(b.) Then, run the fairtiny teacher step:
```
./scripts/train_fairtiny_teacher.sh
```

## Acknowledgment and Citation

If you find this repository useful for your research, please consider citing our [paper]

Clement T. Okolo: jebisbal@uc.cl

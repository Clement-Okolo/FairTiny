"""
Baseline knowledge distillation (no fairness term) for 3D OCT volumes (oct_bscans_3d).

3D counterpart of train_baseline_distillation.py. Differences from the 2D script:
  * student and teacher are 2D ImageNet CNNs converted to 3D (Conv3dConverter, i3d weights),
    built exactly like train_dr_fair_real3d.py / train_dr_fair_real3d_fis.py build them, so a
    teacher checkpoint from either of those scripts loads directly;
  * one GPU per run (choose it with CUDA_VISIBLE_DEVICES), no DataParallel;
  * no ViT-B branch (the 3D pipeline has no 3D ViT).

Used by train_real3d_baseline_distillation.sh (teacher = 3D baseline checkpoint from train_dr_fair_real3d.py).
"""
import os
import sys
import argparse
import random
import time
import json
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

# make "src" importable no matter where the script is launched from (repo root = parent of scripts/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.modules import *
from src.data_handler import *
from src import logger
from typing import NamedTuple

from fairlearn.metrics import *

from src.utils.utils import model_to_syncbn, Transform3D
from src.acsconv.converters import ACSConverter, Conv3dConverter, Conv2_5dConverter

SUPPORTED_3D_MODELS = ['mobilenet', 'efficientnet', 'densenet', 'resnet', 'vgg']


class Dataset_Info(NamedTuple):
    beta: float = 0.9999
    gamma: float = 2.0
    samples_per_attr: list[int] = [0, 0, 0]
    loss_type: str = "sigmoid"
    no_of_classes: int = 2
    no_of_attr: int = 3


# CUDA_VISIBLE_DEVICES (set in the .sh) exposes exactly one GPU to each run
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='3D baseline knowledge distillation')

parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, dest='lr')
parser.add_argument('--momentum', default=0.1, type=float)
parser.add_argument('--wd', '--weight-decay', default=0.01, type=float, dest='weight_decay')
parser.add_argument('--seed', default=13, type=int)
parser.add_argument('--pretrained-weights', default='', type=str, help='resume from a checkpoint of this script')
parser.add_argument('--result_dir', default='./results', type=str)
parser.add_argument('--data_dir', default='.', type=str)
parser.add_argument('--model_type', default='mobilenet', type=str, help='|'.join(SUPPORTED_3D_MODELS))
parser.add_argument('--task', default='cls', type=str)
parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--loss_type', default='bce', type=str)
parser.add_argument('--modality_types', default='oct_bscans_3d', type=str)
parser.add_argument('--perf_file', default='', type=str)
parser.add_argument('--attribute_type', default='race', type=str,
                    help='race|gender|hispanic|maritalstatus|language|genderrace|genderethnicity')
parser.add_argument('--need_balance', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--dataset_proportion', default=1., type=float)
parser.add_argument('--num_classes', default=2, type=int)
parser.add_argument('--bootstrap_repeat_times', default=100, type=int)
parser.add_argument('--conv_type', default='Conv3d', type=str, help='Conv3d|ACSConv|Conv2_5d')
parser.add_argument('--pretrained_3d', default='i3d', type=str)
parser.add_argument('--shape_transform', action='store_true')

# knowledge distillation
parser.add_argument('--teacher_model_path', type=str, required=True,
                    help='model_best_epoch.pth from train_dr_fair_real3d.py or train_dr_fair_real3d_fis.py')
parser.add_argument('--teacher_model_type', type=str, required=True, help='|'.join(SUPPORTED_3D_MODELS))
parser.add_argument('--teacher_conv_type', type=str, default=None,
                    help='conv type the teacher was trained with (default: same as --conv_type)')
parser.add_argument('--distill_alpha', type=float, default=0.5,
                    help='weight of the soft (teacher) loss; hard loss gets 1 - alpha')
parser.add_argument('--distill_temp', type=float, default=2.0)

ATTRIBUTE_TO_INDEX = {'race': 0, 'gender': 1, 'hispanic': 2, 'maritalstatus': 3,
                      'language': 4, 'genderrace': 5, 'genderethnicity': 6}
ATTRIBUTE_TO_NUM_GROUPS = {'race': 3, 'gender': 2, 'hispanic': 2, 'maritalstatus': 5,
                           'language': 3, 'genderrace': 5, 'genderethnicity': 4}


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)


# ----------------------------------------------------------------------------------------------
# 3D model construction (identical to train_dr_fair_real3d.py so checkpoints are interchangeable)
# ----------------------------------------------------------------------------------------------
def mobilenet_forward_3d(self, x):
    # torchvision MobileNetV2.forward, but the final average is over depth, height and width
    x = self.features(x)
    x = F.adaptive_avg_pool3d(x, 1)
    x = torch.flatten(x, 1)
    return self.classifier(x)


def densenet_forward_3d(self, x):
    # torchvision DenseNet.forward, but the final average is over depth, height and width
    x = self.features(x)
    x = F.relu(x)
    x = F.adaptive_avg_pool3d(x, 1)
    x = torch.flatten(x, 1)
    return self.classifier(x)


def build_3d_model(model_type, out_dim, conv_type='Conv3d', pretrained_3d='i3d'):
    if model_type not in SUPPORTED_3D_MODELS:
        raise NotImplementedError(f'{model_type} is not supported for oct_bscans_3d; use one of {SUPPORTED_3D_MODELS}')

    model = create_model(model_type=model_type, in_dim=1, out_dim=out_dim)  # ImageNet-pretrained 2D model
    if model_type == 'mobilenet':
        model.forward = types.MethodType(mobilenet_forward_3d, model)
    elif model_type == 'densenet':
        model.forward = types.MethodType(densenet_forward_3d, model)

    if conv_type == 'ACSConv':
        model = model_to_syncbn(ACSConverter(model))
    elif conv_type == 'Conv2_5d':
        model = model_to_syncbn(Conv2_5dConverter(model))
    elif conv_type == 'Conv3d':
        repeat_axis = -3 if pretrained_3d == 'i3d' else None
        model = model_to_syncbn(Conv3dConverter(model, i3d_repeat_axis=repeat_axis))
    else:
        raise NotImplementedError(f'unknown conv_type {conv_type}')

    if model_type == 'vgg':
        model.avgpool = nn.AdaptiveAvgPool3d((1, 7, 7))

    # .to() returns the underlying nn.Module (the converter only wraps it)
    return model.to(device)


def load_teacher_model(args, out_dim):
    teacher_conv_type = args.teacher_conv_type or args.conv_type
    teacher = build_3d_model(args.teacher_model_type, out_dim, conv_type=teacher_conv_type,
                             pretrained_3d=args.pretrained_3d)
    checkpoint = torch.load(args.teacher_model_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    # strip a DataParallel prefix if the teacher was saved wrapped
    state_dict = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    try:
        teacher.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        raise RuntimeError(f'could not load {args.teacher_model_path} as a 3D {args.teacher_model_type} '
                           f'({teacher_conv_type}). Check that TEACHER_MODEL_TYPE in the .sh matches the model '
                           f'the checkpoint was trained with.\n{str(e)[:500]}') from None
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    logger.log(f'loaded {args.teacher_model_type} teacher from {args.teacher_model_path}')
    if isinstance(checkpoint, dict) and 'test_auc' in checkpoint:
        logger.log(f'teacher checkpoint: epoch {checkpoint.get("epoch")}, test AUC {checkpoint["test_auc"]:.4f}')
    return teacher


# ----------------------------------------------------------------------------------------------
# losses
# ----------------------------------------------------------------------------------------------
def distillation_loss(student_logits, teacher_logits, target, criterion, alpha=0.5, temperature=2.0):
    """Same as src/knowledge_distillation.py in FairTiny (kept here so the script has no extra dependency)."""
    hard_loss = criterion(student_logits, target)

    if student_logits.dim() == 1 or student_logits.shape[1] == 1:
        student_logits = student_logits.view(-1)
        teacher_logits = teacher_logits.view(-1)
        teacher_probs = torch.sigmoid(teacher_logits / temperature)
        student_log_probs = F.logsigmoid(student_logits / temperature)
        student_log_one_minus_probs = F.logsigmoid(-student_logits / temperature)
        soft_loss = -(teacher_probs * student_log_probs +
                      (1 - teacher_probs) * student_log_one_minus_probs).mean() * (temperature ** 2)
    else:
        soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
        soft_student = F.log_softmax(student_logits / temperature, dim=1)
        soft_loss = -(soft_teacher * soft_student).sum(dim=1).mean() * (temperature ** 2)

    total_loss = (1 - alpha) * hard_loss + alpha * soft_loss
    return total_loss, hard_loss, soft_loss


# ----------------------------------------------------------------------------------------------
# train / eval
# ----------------------------------------------------------------------------------------------
def train_distill(model, teacher_model, optimizer, scaler, loader, epoch, num_classes=2, args=None):
    model.train()
    teacher_model.eval()

    loss_batch, hard_loss_batch, soft_loss_batch = [], [], []
    preds, gts, attrs = [], [], []
    skipped = 0

    if num_classes == 2:
        hard_criterion = nn.BCEWithLogitsLoss()
    else:
        hard_criterion = nn.CrossEntropyLoss()

    t1 = time.time()
    for i, (input, target, attr) in enumerate(loader):
        input = input.to(device)
        target = target.to(device)

        with torch.cuda.amp.autocast():
            student_pred = model(input)
            with torch.no_grad():
                teacher_pred = teacher_model(input)

            if not torch.isfinite(student_pred).all():
                skipped += 1
                print(f'epoch {epoch} iter {i}: student output is NaN/inf, skipping this batch', flush=True)
                optimizer.zero_grad()
                continue

            if num_classes == 2:
                student_pred = student_pred.squeeze(1)
                teacher_pred = teacher_pred.squeeze(1)
                loss, hard_loss, soft_loss = distillation_loss(
                    student_pred, teacher_pred.float(), target, hard_criterion,
                    alpha=args.distill_alpha, temperature=args.distill_temp)
                pred_prob = torch.sigmoid(student_pred.detach())
            else:
                loss, hard_loss, soft_loss = distillation_loss(
                    student_pred, teacher_pred.float(), target.long(), hard_criterion,
                    alpha=args.distill_alpha, temperature=args.distill_temp)
                pred_prob = F.softmax(student_pred.detach(), dim=1)

        preds.append(pred_prob.float().cpu().numpy())
        gts.append(target.detach().cpu().numpy())
        attrs.append(torch.vstack(attr).cpu().numpy())

        loss_batch.append(loss.item())
        hard_loss_batch.append(hard_loss.mean().item())
        soft_loss_batch.append(soft_loss.mean().item())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if i % 100 == 0:
            print(f'epoch {epoch} iter {i}: loss {loss.item():.4f} hard {hard_loss.mean().item():.4f} '
                  f'soft {soft_loss.mean().item():.4f}', flush=True)

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    attrs = np.concatenate(attrs, axis=1).astype(int)

    cur_auc = compute_auc(preds, gts, num_classes=num_classes)
    if num_classes == 2:
        acc = accuracy(preds, gts, topk=(1,))
    else:
        acc = accuracy(torch.from_numpy(preds), torch.from_numpy(gts), topk=(1,))

    print(f'train ====> epoch {epoch} loss: {np.mean(loss_batch):.4f} hard: {np.mean(hard_loss_batch):.4f} '
          f'soft: {np.mean(soft_loss_batch):.4f} auc: {cur_auc:.4f} skipped: {skipped} time: {time.time() - t1:.1f}s')

    return np.mean(loss_batch), acc, cur_auc, preds, gts, attrs


def validation(model, criterion, loader, epoch, num_classes=2):
    model.eval()

    loss_batch = []
    preds, gts, attrs = [], [], []

    with torch.no_grad():
        for i, (input, target, attr) in enumerate(loader):
            input = input.to(device)
            target = target.to(device)

            pred = model(input)
            if pred.shape[1] == 1:
                pred = pred.squeeze(1)
                loss = criterion(pred, target).mean()
                pred_prob = torch.sigmoid(pred)
            else:
                loss = criterion(pred, target.long()).mean()
                pred_prob = F.softmax(pred, dim=1)

            preds.append(pred_prob.cpu().numpy())
            gts.append(target.cpu().numpy())
            attrs.append(torch.vstack(attr).cpu().numpy())
            loss_batch.append(loss.item())

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    attrs = np.concatenate(attrs, axis=1).astype(int)

    cur_auc = compute_auc(preds, gts, num_classes=num_classes)
    if num_classes == 2:
        acc = accuracy(preds, gts, topk=(1,))
    else:
        acc = accuracy(torch.from_numpy(preds), torch.from_numpy(gts), topk=(1,))

    print(f'test <==== epoch {epoch} loss: {np.mean(loss_batch):.4f} auc: {cur_auc:.4f}')
    return np.mean(loss_batch), acc, cur_auc, preds, gts, attrs


def log_split_metrics(prefix, es_acc, es_auc, aucs_by_attrs, between_group_disparity, dpds, eods):
    for ii in range(len(es_acc)):
        logger.logkv(f'{prefix}_es_acc_attr{ii}', round(es_acc[ii], 4))
    for ii in range(len(es_auc)):
        logger.logkv(f'{prefix}_es_auc_attr{ii}', round(es_auc[ii], 4))
    for ii in range(len(aucs_by_attrs)):
        for iii in range(len(aucs_by_attrs[ii])):
            logger.logkv(f'{prefix}_auc_attr{ii}_group{iii}', round(aucs_by_attrs[ii][iii], 4))
    for ii in range(len(between_group_disparity)):
        logger.logkv(f'{prefix}_auc_attr{ii}_std_group_disparity', round(between_group_disparity[ii][0], 4))
        logger.logkv(f'{prefix}_auc_attr{ii}_max_group_disparity', round(between_group_disparity[ii][1], 4))
    for ii in range(len(dpds)):
        logger.logkv(f'{prefix}_dpd_attr{ii}', round(dpds[ii], 4))
    for ii in range(len(eods)):
        logger.logkv(f'{prefix}_eod_attr{ii}', round(eods[ii], 4))


if __name__ == '__main__':
    args = parser.parse_args()

    if args.modality_types != 'oct_bscans_3d':
        raise ValueError('this script is for --modality_types oct_bscans_3d; use train_baseline_distillation.py for 2D')

    if args.seed < 0:
        args.seed = int(np.random.randint(10000, size=1)[0])
    set_random_seed(args.seed)

    logger.configure(dir=args.result_dir, log_suffix='train_distill')
    logger.log(f'===> random seed: {args.seed}')
    with open(os.path.join(args.result_dir, 'args_train_distill.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    train_transform = Transform3D(mul='random') if args.shape_transform else Transform3D()
    eval_transform = Transform3D(mul='0.5') if args.shape_transform else Transform3D()

    train_dataset = Harvard_DR_Fairness(os.path.join(args.data_dir, 'train'), modality_type=args.modality_types, task=args.task,
                                        resolution=args.image_size, attribute_type=args.attribute_type,
                                        needBalance=args.need_balance, transform=train_transform,
                                        dataset_proportion=args.dataset_proportion)
    val_dataset = Harvard_DR_Fairness(os.path.join(args.data_dir, 'val'), modality_type=args.modality_types, task=args.task,
                                      resolution=args.image_size, transform=eval_transform, attribute_type=args.attribute_type)
    test_dataset = Harvard_DR_Fairness(os.path.join(args.data_dir, 'test'), modality_type=args.modality_types, task=args.task,
                                       resolution=args.image_size, transform=eval_transform, attribute_type=args.attribute_type)

    args.num_classes = int(np.max(list(train_dataset.disease_mapping.values()))) + 1
    logger.log(f'there are {args.num_classes} classes in total')
    logger.log(train_dataset.disease_mapping)
    logger.log(f'train {len(train_dataset)} samples, val {len(val_dataset)} samples, test {len(test_dataset)} samples')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True, drop_last=False)

    samples_per_attr = get_num_by_group_(train_loader)
    logger.log('group information:')
    logger.log(samples_per_attr)
    ds_info = Dataset_Info(no_of_attr=len(samples_per_attr))

    best_global_perf_file = os.path.join(os.path.dirname(args.result_dir), f'best_{args.perf_file}')
    if args.perf_file != '' and not os.path.exists(best_global_perf_file):
        auc_head_str = ''
        for i in range(len(samples_per_attr)):
            auc_head_str += ', '.join([f'auc_attr{i}_group{x}' for x in range(len(samples_per_attr[i]))]) + ', '
        dpd_head_str = ', '.join([f'dpd_attr{i}' for i in range(len(samples_per_attr))]) + ', '
        eod_head_str = ', '.join([f'eod_attr{i}' for i in range(len(samples_per_attr))]) + ', '
        esacc_head_str = ', '.join([f'esacc_attr{i}' for i in range(len(samples_per_attr))]) + ', '
        esauc_head_str = ', '.join([f'esauc_attr{i}' for i in range(len(samples_per_attr))]) + ', '
        group_disparity_head_str = ', '.join([f'std_group_disparity_attr{i}, max_group_disparity_attr{i}'
                                              for i in range(len(samples_per_attr))]) + ', '
        with open(best_global_perf_file, 'w') as f:
            f.write(f'epoch, acc, {esacc_head_str} auc, {esauc_head_str} {auc_head_str} {dpd_head_str} '
                    f'{eod_head_str} {group_disparity_head_str} path\n')

    if args.num_classes == 2:
        out_dim = 1
        criterion = nn.BCEWithLogitsLoss()
    else:
        out_dim = args.num_classes
        criterion = nn.CrossEntropyLoss()

    # student
    model = build_3d_model(args.model_type, out_dim, conv_type=args.conv_type, pretrained_3d=args.pretrained_3d)
    logger.log(f'student: 3D {args.model_type} ({args.conv_type})')

    # teacher
    teacher_model = load_teacher_model(args, out_dim)

    scaler = torch.cuda.amp.GradScaler()
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.0, 0.1), weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    start_epoch = 0
    if args.pretrained_weights != '':
        checkpoint = torch.load(args.pretrained_weights, map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    best_auc = sys.float_info.min
    best_ep = 0
    best_auc_groups = None

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc, train_auc, _, _, _ = train_distill(
            model, teacher_model, optimizer, scaler, train_loader, epoch,
            num_classes=args.num_classes, args=args)
        val_loss, val_acc, val_auc, val_preds, val_gts, val_attrs = validation(
            model, criterion, val_loader, epoch, num_classes=args.num_classes)
        test_loss, test_acc, test_auc, test_preds, test_gts, test_attrs = validation(
            model, criterion, test_loader, epoch, num_classes=args.num_classes)
        scheduler.step()

        val_es_acc, val_es_auc, val_aucs_by_attrs, val_dpds, val_eods, val_between_group_disparity = \
            evalute_comprehensive_perf(val_preds, val_gts, val_attrs, num_classes=args.num_classes)
        test_es_acc, test_es_auc, test_aucs_by_attrs, test_dpds, test_eods, test_between_group_disparity = \
            evalute_comprehensive_perf(test_preds, test_gts, test_attrs, num_classes=args.num_classes)

        # same selection rule as the other FairTiny / FAS scripts (best test AUC)
        if test_auc > best_auc:
            best_auc = test_auc
            best_acc = test_acc
            best_ep = epoch
            best_auc_groups = test_aucs_by_attrs
            best_dpd_groups = test_dpds
            best_eod_groups = test_eods
            best_es_acc = test_es_acc
            best_es_auc = test_es_auc
            best_between_group_disparity = test_between_group_disparity
            best_test_preds, best_test_gts, best_test_attrs = test_preds, test_gts, test_attrs

            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_auc': train_auc,
                'val_auc': val_auc,
                'test_auc': test_auc,
            }
            torch.save(state, os.path.join(args.result_dir, 'model_best_distill_epoch.pth'))
            np.savez(os.path.join(args.result_dir, 'pred_gt_best_distill_epoch.npz'),
                     test_pred=test_preds, test_gt=test_gts, test_attr=test_attrs)

        logger.log(f'---- best AUC {best_auc:.4f} at epoch {best_ep}')
        logger.log(f'---- best AUC by groups and attributes at epoch {best_ep}')
        logger.log(best_auc_groups)

        logger.logkv('epoch', epoch)
        logger.logkv('train_loss', round(train_loss, 4))
        logger.logkv('train_acc', round(train_acc, 4))
        logger.logkv('train_auc', round(train_auc, 4))
        logger.logkv('val_loss', round(val_loss, 4))
        logger.logkv('val_acc', round(val_acc, 4))
        logger.logkv('val_auc', round(val_auc, 4))
        logger.logkv('test_loss', round(test_loss, 4))
        logger.logkv('test_acc', round(test_acc, 4))
        logger.logkv('test_auc', round(test_auc, 4))
        log_split_metrics('val', val_es_acc, val_es_auc, val_aucs_by_attrs, val_between_group_disparity, val_dpds, val_eods)
        log_split_metrics('test', test_es_acc, test_es_auc, test_aucs_by_attrs, test_between_group_disparity, test_dpds, test_eods)
        logger.dumpkvs()

    final_dir = f'{args.result_dir}_seed{args.seed}_auc{best_auc:.4f}_distill'

    if args.perf_file != '' and os.path.exists(best_global_perf_file):
        with open(best_global_perf_file, 'a') as f:
            esacc_str = ', '.join([f'{x:.4f}' for x in best_es_acc]) + ', '
            esauc_str = ', '.join([f'{x:.4f}' for x in best_es_auc]) + ', '
            auc_str = ''
            for i in range(len(best_auc_groups)):
                auc_str += ', '.join([f'{x:.4f}' for x in best_auc_groups[i]]) + ', '
            disparity_str = ''
            for i in range(len(best_between_group_disparity)):
                disparity_str += ', '.join([f'{x:.4f}' for x in best_between_group_disparity[i]]) + ', '
            dpd_str = ', '.join([f'{x:.4f}' for x in best_dpd_groups]) + ', '
            eod_str = ', '.join([f'{x:.4f}' for x in best_eod_groups]) + ', '
            f.write(f'{best_ep}, {best_acc:.4f}, {esacc_str} {best_auc:.4f}, {esauc_str} {auc_str} {dpd_str} '
                    f'{eod_str} {disparity_str} {final_dir}\n')

    best_acc, best_es_acc, best_auc, best_es_auc, best_auc_groups, best_dpd_groups, best_eod_groups, best_between_group_disparity, \
        best_acc_std, best_es_acc_std, best_auc_std, best_es_auc_std, best_auc_groups_std, best_dpd_groups_std, best_eod_groups_std, \
        best_between_group_disparity_std = bootstrap_performance(best_test_preds, best_test_gts, best_test_attrs,
                                                                 num_classes=args.num_classes,
                                                                 bootstrap_repeat_times=args.bootstrap_repeat_times)
    logger.log('(mean) best_acc, best_es_acc, best_auc, best_es_auc, best_auc_groups, best_dpd_groups, best_eod_groups, best_between_group_disparity')
    logger.log(best_acc, best_es_acc, best_auc, best_es_auc, best_auc_groups, best_dpd_groups, best_eod_groups, best_between_group_disparity)
    logger.log('(std) best_acc, best_es_acc, best_auc, best_es_auc, best_auc_groups, best_dpd_groups, best_eod_groups, best_between_group_disparity')
    logger.log(best_acc_std, best_es_acc_std, best_auc_std, best_es_auc_std, best_auc_groups_std, best_dpd_groups_std, best_eod_groups_std, best_between_group_disparity_std)

    os.rename(args.result_dir, final_dir)

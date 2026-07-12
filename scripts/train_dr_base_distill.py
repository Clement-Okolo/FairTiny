import os
import argparse
import random
import time
import json

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import *
from torch.optim import *
import torch.nn.functional as F

from sklearn.metrics import *
from sklearn.model_selection import KFold
from sklearn.utils import *

import sys
sys.path.append('/medailab/medailab/clement/FairAdaptiveScaling')

from src.modules import *
from src.data_handler import *
from src import logger
from typing import NamedTuple
from src.knowledge_distillation import distillation_loss

from fairlearn.metrics import *

import models_vit
from models_vit import NativeScalerWithGradNormCount, adjust_learning_rate, param_groups_lrd, interpolate_pos_embed
from timm.models.layers import trunc_normal_
import timm

class Dataset_Info(NamedTuple):
    beta: float = 0.9999
    gamma: float = 2.0
    samples_per_attr: list[int] = [0,0,0]
    loss_type: str = "sigmoid"
    no_of_classes: int = 2
    no_of_attr: int = 3

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='PyTorch Knowledge Distillation Training')

# Common arguments
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=6e-5, type=float,
                    metavar='W', help='weight decay (default: 6e-5)',
                    dest='weight_decay')
parser.add_argument('--seed', default=13, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('--pretrained-weights', default='', type=str)
parser.add_argument('--result_dir', default='./results', type=str)
parser.add_argument('--data_dir', default='.', type=str)
parser.add_argument('--model_type', default='efficientnet', type=str)
parser.add_argument('--task', default='cls', type=str)
parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--loss_type', default='bce', type=str)
parser.add_argument('--modality_types', default='slo_fundus', type=str, help='oct_bscans_3d|slo_fundus')
parser.add_argument('--fuse_coef', default=1.0, type=float)
parser.add_argument('--perf_file', default='', type=str)
parser.add_argument('--attribute_type', default='race', type=str, help='race|gender|hispanic')
parser.add_argument('--subset_name', default='test', type=str)
parser.add_argument("--need_balance", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Oversampling or not")
parser.add_argument('--dataset_proportion', default=1., type=float)
parser.add_argument('--num_classes', default=2, type=int)
parser.add_argument('--bootstrap_repeat_times', default=100, type=int)

# ViT Args
parser.add_argument('--vit_weights', default='imagenet', type=str)
parser.add_argument('--blr', type=float, default=1e-3,
                    help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
parser.add_argument('--min_lr', type=float, default=1e-6,
                    help='lower lr bound for cyclic schedulers that hit 0')
parser.add_argument('--warmup_epochs', type=int, default=5,
                    help='epochs to warmup LR')
parser.add_argument('--weight_decay', type=float, default=0.05,
                    help='weight decay (default: 0.05)')
parser.add_argument('--layer_decay', type=float, default=0.75,
                    help='layer-wise lr decay from ELECTRA/BEiT')
parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: 0.1)')

# Knowledge distillation arguments
parser.add_argument('--teacher_model_path', type=str, required=True,
                    help='Path to the teacher model checkpoint')
parser.add_argument('--teacher_model_type', type=str, required=True,
                    help='the name of the teacher model')
parser.add_argument('--distill_alpha', type=float, default=0.5,
                    help='Weight of distillation loss vs hard-target loss (0.5 = equal weight)')
parser.add_argument('--distill_temp', type=float, default=2.0,
                    help='Temperature for softening probabilities')

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

def train_distill(model, teacher_model, criterion, optimizer, scaler, train_dataset_loader, epoch, total_iteration, dataset_info=None, num_classes=2, args=None):
    """
    Training function with knowledge distillation
    """
    global device

    model.train()
    teacher_model.eval()  # Teacher model is always in eval mode
    
    loss_batch = []
    hard_loss_batch = []
    soft_loss_batch = []
    top1_accuracy_batch = []
    
    preds = []
    gts = []
    attrs = []

    preds_by_attr = [[] for _ in range(dataset_info.no_of_attr)]
    gts_by_attr = [[] for _ in range(dataset_info.no_of_attr)]
    t1 = time.time()
    
    for i, (input, target, attr) in enumerate(train_dataset_loader):
        if args.model_type == 'ViT-B':
            # we use a per iteration (instead of per epoch) lr scheduler
            tmp_lr = adjust_learning_rate(optimizer, i / len(train_dataset_loader) + epoch, args)
            if i == 0:
                print(f"lr: {tmp_lr}")

        with torch.cuda.amp.autocast():
            input = input.to(device)
            target = target.to(device)

            # Forward pass through student model
            student_pred = model(input)
            
            # Forward pass through teacher model (no gradients)
            with torch.no_grad():
                teacher_pred = teacher_model(input)
            
            # Apply the appropriate loss function based on number of classes
            if num_classes == 2:
                # Binary classification case
                if len(student_pred.shape) > 1 and student_pred.shape[1] == 1:
                    student_pred = student_pred.squeeze(1)
                if len(teacher_pred.shape) > 1 and teacher_pred.shape[1] == 1:
                    teacher_pred = teacher_pred.squeeze(1)
                
                loss, hard_loss, soft_loss = distillation_loss(
                    student_pred, teacher_pred, target, criterion, 
                    alpha=args.distill_alpha, temperature=args.distill_temp
                )
                pred_prob = torch.sigmoid(student_pred.detach())
            else:
                # Multi-class classification
                loss, hard_loss, soft_loss = distillation_loss(
                    student_pred, teacher_pred, target.long(), criterion, 
                    alpha=args.distill_alpha, temperature=args.distill_temp
                )
                pred_prob = F.softmax(student_pred.detach(), dim=1)

            preds.append(pred_prob.detach().cpu().numpy())
            gts.append(target.detach().cpu().numpy())
            attr = torch.vstack(attr)
            attrs.append(attr.detach().cpu().numpy())

        loss_batch.append(loss.item())
        hard_loss_batch.append(hard_loss.item())
        soft_loss_batch.append(soft_loss.item())
        
        if num_classes == 2:
            top1_accuracy = accuracy(pred_prob.detach().cpu().numpy(), target.detach().cpu().numpy(), topk=(1,))
        elif num_classes > 2:
            top1_accuracy = accuracy(pred_prob, target, topk=(1,))
        
        top1_accuracy_batch.append(top1_accuracy)

        if args.model_type == 'ViT-B':
            scaler(loss, optimizer, clip_grad=None,
                        parameters=model.parameters(), create_graph=False,
                        update_grad=True)
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        optimizer.zero_grad()

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    attrs = np.concatenate(attrs, axis=1).astype(int)
    
    cur_auc = compute_auc(preds, gts, num_classes=num_classes)
    if num_classes == 2:
        acc = accuracy(preds, gts, topk=(1,))
    elif num_classes > 2:
        acc = accuracy(torch.from_numpy(preds).cuda(), torch.from_numpy(gts).cuda(), topk=(1,))

    torch.cuda.synchronize()
    t2 = time.time()

    print(f"train ====> epoch {epoch} loss: {np.mean(loss_batch):.4f} hard: {np.mean(hard_loss_batch):.4f} "
          f"soft: {np.mean(soft_loss_batch):.4f} auc: {cur_auc:.4f} time: {t2 - t1:.4f}")

    t1 = time.time()

    return np.mean(loss_batch), acc, cur_auc, preds, gts, attrs

def validation(model, criterion, optimizer, validation_dataset_loader, epoch, result_dir=None, dataset_info=None, num_classes=2):
    """
    Validation function (same as original)
    """
    global device

    model.eval()
    
    loss_batch = []
    top1_accuracy_batch = []
    top5_accuracy_batch = []

    preds = []
    gts = []
    attrs = []
    datadirs = []

    preds_by_attr = [ [] for _ in range(dataset_info.no_of_attr) ]
    gts_by_attr = [ [] for _ in range(dataset_info.no_of_attr) ]

    with torch.no_grad():
        for i, (input, target, attr) in enumerate(validation_dataset_loader):
            input = input.to(device)
            target = target.to(device)
            
            pred = model(input) # .squeeze(1)

            if pred.shape[1] == 1:
                pred = pred.squeeze(1)
                loss = criterion(pred, target)
                pred_prob = torch.sigmoid(pred.detach())
            elif pred.shape[1] > 1:
                loss = criterion(pred, target.long()).mean()
                pred_prob = F.softmax(pred.detach(), dim=1)

            preds.append(pred_prob.detach().cpu().numpy())
            gts.append(target.detach().cpu().numpy())
            attr = torch.vstack(attr)
            attrs.append(attr.detach().cpu().numpy())
            

            loss_batch.append(loss.item())

            if num_classes == 2:
                top1_accuracy = accuracy(pred_prob.detach().cpu().numpy(), target.detach().cpu().numpy(), topk=(1,))
            elif num_classes > 2:
                top1_accuracy = accuracy(pred_prob, target, topk=(1,))
        
            top1_accuracy_batch.append(top1_accuracy)
        
    loss = np.mean(loss_batch)

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    attrs = np.concatenate(attrs, axis=1).astype(int)
    
    cur_auc = compute_auc(preds, gts, num_classes=num_classes)

    if num_classes == 2:
        acc = accuracy(preds, gts, topk=(1,))
    elif num_classes > 2:
        acc = accuracy(torch.from_numpy(preds).cuda(), torch.from_numpy(gts).cuda(), topk=(1,))

    print(f"test <==== epoch {epoch} loss: {np.mean(loss_batch):.4f} auc: {cur_auc:.4f}")

    return loss, acc, cur_auc, preds, gts, attrs

def load_teacher_model(model_path, args):
    """
    Load a pre-trained teacher model
    """
    # Create model with same structure as in original training
    if args.task == 'md':
        out_dim = 1
    elif args.task == 'cls': 
        if args.num_classes == 2:
            out_dim = 1
        elif args.num_classes > 2:
            out_dim = args.num_classes
    elif args.task == 'tds': 
        out_dim = 52
    
    # Create model based on architecture type
    if args.teacher_model_type == 'ViT-B':
        model = models_vit.__dict__["vit_base_patch16"](
            num_classes=out_dim,
            drop_path_rate=args.drop_path,
            global_pool=True,
        )
        
        if args.modality_types == 'oct_bscans':
            model.patch_embed.proj = nn.Conv2d(128, 768, kernel_size=(16, 16), stride=(16, 16))
    else:
        # For other model types, use create_model function
        if args.modality_types == 'slo_fundus':
            in_dim = 3
        elif args.modality_types == 'oct_bscans':
            in_dim = 128
        elif args.modality_types == 'oct_bscans_3d':
            in_dim = 1
        elif args.modality_types == 'rnflt+ilm':
            in_dim = 2
        elif args.modality_types == 'ilm' or args.modality_types == 'rnflt':
            in_dim = 1
            
        model = create_model(model_type=args.teacher_model_type, in_dim=in_dim, out_dim=out_dim)
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    print(f"Loaded teacher model from {model_path}")
    return model

if __name__ == '__main__':
    args = parser.parse_args()

    if args.seed < 0:
        args.seed = int(np.random.randint(10000, size=1)[0])
    set_random_seed(args.seed)

    logger.log(f'===> random seed: {args.seed}')

    logger.configure(dir=args.result_dir, log_suffix='train_distill')

    with open(os.path.join(args.result_dir, f'args_train_distill.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.model_type == 'vit' or args.model_type == 'swin' or args.model_type == 'ViT-B':
        args.image_size = 224

    # Load datasets
    train_havo_dataset = Harvard_DR_Fairness(os.path.join(args.data_dir, 'train'), modality_type=args.modality_types, task=args.task, resolution=args.image_size, attribute_type=args.attribute_type, needBalance=args.need_balance, dataset_proportion=args.dataset_proportion)
    val_havo_dataset = Harvard_DR_Fairness(os.path.join(args.data_dir, 'val'), modality_type=args.modality_types, task=args.task, resolution=args.image_size, attribute_type=args.attribute_type)
    test_havo_dataset = Harvard_DR_Fairness(os.path.join(args.data_dir, 'test'), modality_type=args.modality_types, task=args.task, resolution=args.image_size, attribute_type=args.attribute_type)

    args.num_classes = int(np.max(list(train_havo_dataset.disease_mapping.values())))+1
    logger.log(f'there are {args.num_classes} classes in total')
    logger.log(train_havo_dataset.disease_mapping)

    logger.log(f'train patients {len(train_havo_dataset)} with {len(train_havo_dataset)} samples, val patients {len(val_havo_dataset)} with {len(val_havo_dataset)} samples, test patients {len(test_havo_dataset)} with {len(test_havo_dataset)} samples')

    # Create data loaders
    train_dataset_loader = torch.utils.data.DataLoader(
        train_havo_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    
    validation_dataset_loader = torch.utils.data.DataLoader(
        val_havo_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    test_dataset_loader = torch.utils.data.DataLoader(
        test_havo_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)
    
    # Get information about groups for fairness
    print(len(train_dataset_loader))
    
    samples_per_attr = get_num_by_group_(train_dataset_loader)
    logger.log(f'group information:')
    logger.log(samples_per_attr)
    ds_info = Dataset_Info(no_of_attr=len(samples_per_attr))

    # Configure output directories and files for tracking results
    best_global_perf_file = os.path.join(os.path.dirname(args.result_dir), f'best_{args.perf_file}')
    lastep_global_perf_file = os.path.join(os.path.dirname(args.result_dir), f'last_{args.perf_file}')

    # Prepare CSV headers for tracking performance
    acc_head_str = ''
    auc_head_str = ''
    dpd_head_str = ''
    eod_head_str = ''
    esacc_head_str = ''
    esauc_head_str = ''
    group_disparity_head_str = ''
    if args.perf_file != '':
        if not os.path.exists(best_global_perf_file):
            for i in range(len(samples_per_attr)):
                auc_head_str += ', '.join([f'auc_attr{i}_group{x}' for x in range(len(samples_per_attr[i]))]) + ', '
            dpd_head_str += ', '.join([f'dpd_attr{i}' for i in range(len(samples_per_attr))]) + ', '
            eod_head_str += ', '.join([f'eod_attr{i}' for i in range(len(samples_per_attr))]) + ', '
            esacc_head_str += ', '.join([f'esacc_attr{i}' for i in range(len(samples_per_attr))]) + ', '
            esauc_head_str += ', '.join([f'esauc_attr{i}' for i in range(len(samples_per_attr))]) + ', '

            group_disparity_head_str += ', '.join([f'std_group_disparity_attr{i}, max_group_disparity_attr{i}' for i in range(len(samples_per_attr))]) + ', '
            
            with open(best_global_perf_file, 'w') as f:
                f.write(f'epoch, acc, {esacc_head_str} auc, {esauc_head_str} {auc_head_str} {dpd_head_str} {eod_head_str} {group_disparity_head_str} path\n')

    # Set up output dimensions based on task
    if args.task == 'md':
        out_dim = 1
        criterion = nn.MSELoss()
        predictor_head = nn.Identity()
    elif args.task == 'cls': 
        out_dim = 1
        if args.num_classes == 2:
            out_dim = 1
            criterion = nn.BCEWithLogitsLoss()
        elif args.num_classes > 2:
            out_dim = args.num_classes
            criterion = nn.CrossEntropyLoss()
        predictor_head = nn.Sigmoid()
    elif args.task == 'tds': 
        out_dim = 52
        criterion = nn.MSELoss()
        predictor_head = nn.Identity()

    # Determine input dimensions based on modality
    if args.modality_types == 'slo_fundus':
        in_dim = 3
    elif args.modality_types == 'oct_bscans':
        in_dim = 128
    elif args.modality_types == 'oct_bscans_3d':
        in_dim = 1
    elif args.modality_types == 'rnflt+ilm':
        in_dim = 2
    elif args.modality_types == 'ilm' or args.modality_types == 'rnflt':
        in_dim = 1

    # Create student model
    if args.model_type == 'ViT-B':
        args.lr = args.blr * args.batch_size / 256
        args.image_size = 224

        model = models_vit.__dict__["vit_base_patch16"](
            num_classes=out_dim,
            drop_path_rate=args.drop_path,
            global_pool=True,
        )
        
        if args.modality_types == 'oct_bscans':
            model.patch_embed.proj = nn.Conv2d(128, 768, kernel_size=(16, 16), stride=(16, 16))
        
        # Manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)
        
        # Build optimizer with layer-wise lr decay (lrd)
        no_weight_decay_list = model.no_weight_decay()
        
        param_groups = param_groups_lrd(model, args.weight_decay,
            no_weight_decay_list=no_weight_decay_list,
            layer_decay=args.layer_decay
        )
    
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
        scaler = NativeScalerWithGradNormCount()
    else:
        # For other model types
        scaler = torch.cuda.amp.GradScaler()
        model = create_model(model_type=args.model_type, in_dim=in_dim, out_dim=out_dim)
        optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.0, 0.1), weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    model = model.to(device)

    # Load the teacher model
    teacher_model = load_teacher_model(args.teacher_model_path, args)
    
    # Initialize variables for training
    start_epoch = 0
    best_top1_accuracy = 0.

    # Resume from checkpoint if provided
    if args.pretrained_weights != "":
        checkpoint = torch.load(args.pretrained_weights)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if args.model_type != 'ViT-B':
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    total_iteration = len(train_havo_dataset)//args.batch_size

    # Performance tracking variables
    best_auc_groups = None
    best_acc_groups = None
    best_pred_gt_by_attr = None
    best_auc = sys.float_info.min
    best_acc = sys.float_info.min
    best_es_acc = sys.float_info.min
    best_es_auc = sys.float_info.min
    best_ep = 0
    best_between_group_disparity = None
    best_val_auc = sys.float_info.min

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Train with knowledge distillation
        train_loss, train_acc, train_auc, train_preds, train_gts, train_attrs = train_distill(
            model, teacher_model, criterion, optimizer, scaler, train_dataset_loader, 
            epoch, total_iteration, dataset_info=ds_info, num_classes=args.num_classes, args=args
        )
        
        # Validation
        val_loss, val_acc, val_auc, val_preds, val_gts, val_attrs = validation(
            model, criterion, optimizer, validation_dataset_loader, 
            epoch, dataset_info=ds_info, num_classes=args.num_classes
        )
        
        # Test
        test_loss, test_acc, test_auc, test_preds, test_gts, test_attrs = validation(
            model, criterion, optimizer, test_dataset_loader, 
            epoch, dataset_info=ds_info, num_classes=args.num_classes
        )
        
        # Update learning rate scheduler
        if args.model_type != 'ViT-B': # we use per-iter lr scheduler for ViT
            scheduler.step()

        # Calculate fairness metrics
        val_es_acc, val_es_auc, val_aucs_by_attrs, val_dpds, val_eods, val_between_group_disparity = evalute_comprehensive_perf(
            val_preds, val_gts, val_attrs, num_classes=args.num_classes
        )
        
        test_es_acc, test_es_auc, test_aucs_by_attrs, test_dpds, test_eods, test_between_group_disparity = evalute_comprehensive_perf(
            test_preds, test_gts, test_attrs, num_classes=args.num_classes
        )

        # Save best model
        if test_auc > best_val_auc:
            best_val_auc = test_auc
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

            if args.model_type == 'ViT-B':
                state = {
                'epoch': epoch,# zero indexing
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'scaler_state_dict' : scaler.state_dict(),
                # 'scheduler_state_dict' : scheduler.state_dict(),
                'train_auc': train_auc,
                'val_auc': val_auc,
                'test_auc': test_auc
                }
            else:
                state = {
                'epoch': epoch,# zero indexing
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'scaler_state_dict' : scaler.state_dict(),
                'scheduler_state_dict' : scheduler.state_dict(),
                'train_auc': train_auc,
                'val_auc': val_auc,
                'test_auc': test_auc
                }
            torch.save(state, os.path.join(args.result_dir, f"model_best_distill_epoch.pth"))

            if args.result_dir is not None:
                np.savez(os.path.join(args.result_dir, f'pred_gt_best_distill_epoch.npz'), 
                            test_pred=test_preds, test_gt=test_gts, test_attr=test_attrs)

        logger.log(f'---- best val AUC {best_val_auc:.4f} at epoch {best_ep}')
        logger.log(f'---- best AUC {best_auc:.4f} at epoch {best_ep}')
        logger.log(f'---- best AUC by groups and attributes at epoch {best_ep}')
        logger.log(best_auc_groups)

        logger.logkv('epoch', epoch)
        logger.logkv('train_loss', round(train_loss,4))
        logger.logkv('train_acc', round(train_acc,4))
        logger.logkv('train_auc', round(train_auc,4))

        logger.logkv('val_loss', round(val_loss,4))
        logger.logkv('val_acc', round(val_acc,4))
        logger.logkv('val_auc', round(val_auc,4))

        logger.logkv('test_loss', round(test_loss,4))
        logger.logkv('test_acc', round(test_acc,4))
        logger.logkv('test_auc', round(test_auc,4))

        for ii in range(len(val_es_acc)):
            logger.logkv(f'val_es_acc_attr{ii}', round(val_es_acc[ii],4))
        for ii in range(len(val_es_auc)):
            logger.logkv(f'val_es_auc_attr{ii}', round(val_es_auc[ii],4))
        for ii in range(len(val_aucs_by_attrs)):
            for iii in range(len(val_aucs_by_attrs[ii])):
                logger.logkv(f'val_auc_attr{ii}_group{iii}', round(val_aucs_by_attrs[ii][iii],4))

        for ii in range(len(val_between_group_disparity)):
            logger.logkv(f'val_auc_attr{ii}_std_group_disparity', round(val_between_group_disparity[ii][0],4))
            logger.logkv(f'val_auc_attr{ii}_max_group_disparity', round(val_between_group_disparity[ii][1],4))

        for ii in range(len(val_dpds)):
            logger.logkv(f'val_dpd_attr{ii}', round(val_dpds[ii],4))
        for ii in range(len(val_eods)):
            logger.logkv(f'val_eod_attr{ii}', round(val_eods[ii],4))


        for ii in range(len(test_es_acc)):
            logger.logkv(f'test_es_acc_attr{ii}', round(test_es_acc[ii],4))
        for ii in range(len(test_es_auc)):
            logger.logkv(f'test_es_auc_attr{ii}', round(test_es_auc[ii],4))
        for ii in range(len(test_aucs_by_attrs)):
            for iii in range(len(test_aucs_by_attrs[ii])):
                logger.logkv(f'test_auc_attr{ii}_group{iii}', round(test_aucs_by_attrs[ii][iii],4))

        for ii in range(len(test_between_group_disparity)):
            logger.logkv(f'test_auc_attr{ii}_std_group_disparity', round(test_between_group_disparity[ii][0],4))
            logger.logkv(f'test_auc_attr{ii}_max_group_disparity', round(test_between_group_disparity[ii][1],4))

        for ii in range(len(test_dpds)):
            logger.logkv(f'test_dpd_attr{ii}', round(test_dpds[ii],4))
        for ii in range(len(test_eods)):
            logger.logkv(f'test_eod_attr{ii}', round(test_eods[ii],4))

        logger.dumpkvs()

    if args.perf_file != '':
        if os.path.exists(best_global_perf_file):

            with open(best_global_perf_file, 'a') as f:

                esacc_head_str = ', '.join([f'{x:.4f}' for x in best_es_acc]) + ', '
                esauc_head_str = ', '.join([f'{x:.4f}' for x in best_es_auc]) + ', '

                auc_head_str = ''
                for i in range(len(best_auc_groups)):
                    auc_head_str += ', '.join([f'{x:.4f}' for x in best_auc_groups[i]]) + ', '

                group_disparity_str = ''
                for i in range(len(best_between_group_disparity)):
                    group_disparity_str += ', '.join([f'{x:.4f}' for x in best_between_group_disparity[i]]) + ', '
                
                dpd_head_str = ', '.join([f'{x:.4f}' for x in best_dpd_groups]) + ', '
                eod_head_str = ', '.join([f'{x:.4f}' for x in best_eod_groups]) + ', '

                path_str = f'{args.result_dir}_seed{args.seed}_auc{best_auc:.4f}_distill'
                f.write(f'{best_ep}, {best_acc:.4f}, {esacc_head_str} {best_auc:.4f}, {esauc_head_str} {auc_head_str} {dpd_head_str} {eod_head_str} {group_disparity_str} {path_str}\n')
                
    best_acc, best_es_acc, best_auc, best_es_auc, best_auc_groups, best_dpd_groups, best_eod_groups, best_between_group_disparity, \
    best_acc_std, best_es_acc_std, best_auc_std, best_es_auc_std, best_auc_groups_std, best_dpd_groups_std, best_eod_groups_std, best_between_group_disparity_std = bootstrap_performance(best_test_preds, best_test_gts, best_test_attrs, num_classes=args.num_classes, bootstrap_repeat_times=args.bootstrap_repeat_times)
    logger.log(f'(mean) best_acc, best_es_acc, best_auc, best_es_auc, best_auc_groups, best_dpd_groups, best_eod_groups, best_between_group_disparity')
    logger.log(best_acc, best_es_acc, best_auc, best_es_auc, best_auc_groups, best_dpd_groups, best_eod_groups, best_between_group_disparity)
    logger.log(f'(std) best_acc, best_es_acc, best_auc, best_es_auc, best_auc_groups, best_dpd_groups, best_eod_groups, best_between_group_disparity')
    logger.log(best_acc_std, best_es_acc_std, best_auc_std, best_es_auc_std, best_auc_groups_std, best_dpd_groups_std, best_eod_groups_std, best_between_group_disparity_std)

    os.rename(args.result_dir, f'{args.result_dir}_seed{args.seed}_auc{best_auc:.4f}_distill')
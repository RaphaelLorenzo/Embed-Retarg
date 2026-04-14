import os
import numpy as np
import argparse
import errno
import math
import pickle
from tqdm import tqdm
from time import time
import copy
import random
# import prettytable

import torch
import torch.nn as nn
import shutil
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from lib.utils.tools import *
from lib.utils.learning import *
from lib.data.dataset_embedretarg import EmbedRetargDataset
from lib.data.augmentation import Augmenter2D
from lib.model.loss import *
from tools.conversion_tools import g12h36m_torch

def parse_args():
    parser = argparse.ArgumentParser()
    current_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoints/'+current_timestamp, type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    opts = parser.parse_args()
    return opts

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss):
    print('Saving checkpoint to', chk_path)
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model_pos': model_pos.state_dict(),
        'min_loss' : min_loss
    }, chk_path)
    
        
def train_epoch(args, model_pos, train_loader, train_losses, optimizer, has_gt):
    model_pos.train()
    
    for idx, (batch_input, batch_gt, file, seq_idx) in tqdm(enumerate(train_loader), total=len(train_loader)):    
        
        batch_size = len(batch_input)        
        
        if torch.cuda.is_available():
            batch_input = batch_input.cuda().float()
            batch_gt = batch_gt.cuda().float()
            if batch_gt.shape[2] == 38 and not args.remap_joints_head:
                batch_gt = g12h36m_torch(batch_gt)
            
        with torch.no_grad():
            if args.mask or args.noise:
                batch_input = args.aug.augment2D(batch_input, noise=(args.noise and has_gt), mask=args.mask)
        
        # Predict 3D poses
        predicted_3d_pos = model_pos(batch_input)    # (N, T, TargetJ, 3)
        
        # print(predicted_3d_pos.shape, batch_gt.shape)
        
        optimizer.zero_grad()
        loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt)
        loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)
        loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)
        loss_lv = loss_limb_var(predicted_3d_pos)
        loss_lg = loss_limb_gt(predicted_3d_pos, batch_gt)
        loss_a = loss_angle(predicted_3d_pos, batch_gt)
        loss_av = loss_angle_velocity(predicted_3d_pos, batch_gt)
        loss_total = loss_3d_pos + \
                        args.lambda_scale       * loss_3d_scale + \
                        args.lambda_3d_velocity * loss_3d_velocity + \
                        args.lambda_lv          * loss_lv + \
                        args.lambda_lg          * loss_lg + \
                        args.lambda_a           * loss_a  + \
                        args.lambda_av          * loss_av
        train_losses['3d_pos'].update(loss_3d_pos.item(), batch_size)
        train_losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
        train_losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
        train_losses['lv'].update(loss_lv.item(), batch_size)
        train_losses['lg'].update(loss_lg.item(), batch_size)
        train_losses['angle'].update(loss_a.item(), batch_size)
        train_losses['angle_velocity'].update(loss_av.item(), batch_size)
        train_losses['total'].update(loss_total.item(), batch_size)

        loss_total.backward()

        optimizer.step()
        
def test_epoch(args, model_pos, test_loader, test_losses):
    model_pos.eval()
    
    for idx, (batch_input, batch_gt, file, seq_idx) in tqdm(enumerate(test_loader), total=len(test_loader)):    
        
        batch_size = len(batch_input)        
        
        if torch.cuda.is_available():
            batch_input = batch_input.cuda().float()
            batch_gt = batch_gt.cuda().float()
            if batch_gt.shape[2] == 38 and not args.remap_joints_head:
                batch_gt = g12h36m_torch(batch_gt)
            
        with torch.no_grad():
            # Predict 3D poses without gradient
            predicted_3d_pos = model_pos(batch_input)    # (N, T, TargetJ, 3)
            
        loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt)
        loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)
        loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)
        loss_lv = loss_limb_var(predicted_3d_pos)
        loss_lg = loss_limb_gt(predicted_3d_pos, batch_gt)
        loss_a = loss_angle(predicted_3d_pos, batch_gt)
        loss_av = loss_angle_velocity(predicted_3d_pos, batch_gt)
        loss_total = loss_3d_pos + \
                        args.lambda_scale       * loss_3d_scale + \
                        args.lambda_3d_velocity * loss_3d_velocity + \
                        args.lambda_lv          * loss_lv + \
                        args.lambda_lg          * loss_lg + \
                        args.lambda_a           * loss_a  + \
                        args.lambda_av          * loss_av
        test_losses['3d_pos'].update(loss_3d_pos.item(), batch_size)
        test_losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
        test_losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
        test_losses['lv'].update(loss_lv.item(), batch_size)
        test_losses['lg'].update(loss_lg.item(), batch_size)
        test_losses['angle'].update(loss_a.item(), batch_size)
        test_losses['angle_velocity'].update(loss_av.item(), batch_size)
        test_losses['total'].update(loss_total.item(), batch_size)



def train_with_config(args, opts):
    
    # print config in a readable format
    for key, value in args.items(): 
        if type(value) == list:
            print(f"{key}:")
            for item in value:
                print(f"  - {item}")
        elif type(value) == dict:
            print(f"{key}:")
            for key, value in value.items():
                print(f"  {key}: {value}")
        else:
            print(f"{key}: {value}")
    
    # create checkpoint directory
    try:
        os.makedirs(opts.checkpoint)
        # copy config file to checkpoint directory
        shutil.copy(opts.config, os.path.join(opts.checkpoint, 'config.yaml'))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)


    # load dataset
    print('Loading dataset...')
    trainloader_params = {
          'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 12,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    
    testloader_params = {
          'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 12,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }

    train_dataset = EmbedRetargDataset(data_path=args.data_path,
                                       clip_len=args.clip_len,
                                       stride=args.data_stride,
                                       root_rel_target=args.rootrel,
                                       scale_by=args.scale_by,
                                       scale_range=args.scale_range,
                                       subset='train',
                                       test_keywords=args.test_keywords)
    
    test_dataset = EmbedRetargDataset(data_path=args.data_path,
                                       clip_len=args.clip_len,
                                       stride=args.data_stride,
                                       root_rel_target=args.rootrel,
                                       scale_by=args.scale_by,
                                       scale_range=args.scale_range,
                                       subset='test',
                                       test_keywords=args.test_keywords)
    
    # create data loaders
    train_loader = DataLoader(train_dataset, **trainloader_params)
    test_loader = DataLoader(test_dataset, **testloader_params)
    
    # load modeL and setup for learning
    min_loss = 100000
    args.num_new_joints = 38 if args.remap_joints_head else 0
    model_backbone = load_backbone(args)
    model_params = 0
    for parameter in model_backbone.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()

    # load pretrained checkpoint (or start from scratch)
    if args.finetune:
        print("Finetuning from pretrained checkpoint...")
        chk_filename = args.pretrained_checkpoint
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        if args.remap_joints_head:
            # allow some missing keys because of the new joints
            missing_keys, unexpected_keys = model_backbone.load_state_dict(checkpoint['model_pos'], strict=False)
            print('INFO: Missing keys:', missing_keys)
            assert len(missing_keys) == 2, "We should only have 2 missing keys : 'module.map_to_new_joints.weight', 'module.map_to_new_joints.bias'"
            print('INFO: Unexpected keys:', unexpected_keys)
            assert len(unexpected_keys) == 0, "Unexpected keys found in pretrained checkpoint"
        else:
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
        model_pos = model_backbone            
    else:
        if opts.resume:
            print("Resuming training from latest checkpoint...")
            chk_filename = os.path.join(opts.checkpoint, 'latest_epoch.bin')
            print('Loading checkpoint', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
            model_pos = model_backbone
        else:
            print("Starting training from scratch...")
            model_pos = model_backbone

    # setup optimizer
    lr = args.learning_rate
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_pos.parameters()), lr=lr, weight_decay=args.weight_decay)
    lr_decay = args.lr_decay
    st = 0
    print('INFO: Training on {}(3D) batches'.format(len(train_loader)))
    if opts.resume:
        st = checkpoint['epoch']
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')            
        lr = checkpoint['lr']
        if 'min_loss' in checkpoint and checkpoint['min_loss'] is not None:
            min_loss = checkpoint['min_loss']
    
    # setup augmentation parameters
    args.mask = (args.mask_ratio > 0 and args.mask_T_ratio > 0)
    if args.mask or args.noise:
        args.aug = Augmenter2D(args)
    
    # Training loop
    for epoch in range(st, args.epochs):
        print('Training epoch %d/%d.' % (epoch, args.epochs))
        start_time = time()
        train_losses = {}
        train_losses['3d_pos'] = AverageMeter()
        train_losses['3d_scale'] = AverageMeter()
        train_losses['2d_proj'] = AverageMeter()
        train_losses['lg'] = AverageMeter()
        train_losses['lv'] = AverageMeter()
        train_losses['total'] = AverageMeter()
        train_losses['3d_velocity'] = AverageMeter()
        train_losses['angle'] = AverageMeter()
        train_losses['angle_velocity'] = AverageMeter()

        test_losses = {}
        test_losses['3d_pos'] = AverageMeter()
        test_losses['3d_scale'] = AverageMeter()
        test_losses['2d_proj'] = AverageMeter()
        test_losses['lg'] = AverageMeter()
        test_losses['lv'] = AverageMeter()
        test_losses['total'] = AverageMeter()
        test_losses['3d_velocity'] = AverageMeter()
        test_losses['angle'] = AverageMeter()
        test_losses['angle_velocity'] = AverageMeter()
        N = 0
                    
        # Curriculum Learning
        train_epoch(args, model_pos, train_loader, train_losses, optimizer, has_gt=True) 
        test_epoch(args, model_pos, test_loader, test_losses)
        
        elapsed = (time() - start_time) / 60

        print('[%d] time %.2f lr %f 3d_train %f 3d_test %f' % (
            epoch + 1,
            elapsed,
            lr,
            train_losses['3d_pos'].avg,
            test_losses['3d_pos'].avg))
            
        # Decay learning rate exponentially
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay

        # Save checkpoints
        chk_path = os.path.join(opts.checkpoint, 'epoch_{}.bin'.format(epoch))
        chk_path_latest = os.path.join(opts.checkpoint, 'latest_epoch.bin')
        chk_path_best = os.path.join(opts.checkpoint, 'best_epoch.bin'.format(epoch))
        
        save_checkpoint(chk_path_latest, epoch, lr, optimizer, model_pos, min_loss)
        if (epoch + 1) % args.checkpoint_frequency == 0:
            save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss)
        if test_losses['3d_pos'].avg < min_loss:
            min_loss = test_losses['3d_pos'].avg
            save_checkpoint(chk_path_best, epoch, lr, optimizer, model_pos, min_loss)
                
if __name__ == "__main__":
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    train_with_config(args, opts)
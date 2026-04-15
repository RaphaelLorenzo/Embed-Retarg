import os
import numpy as np
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_embedretarg import EmbedRetargDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pose3d/MB_ft_h36m_global_lite.yaml", help="Path to the config file.")
    parser.add_argument('-e', '--evaluate', default='checkpoints/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-d', '--data_path', type=str, help='data path', default='/home/raphael/Projects/github/accad_subset_random_shapes')
    parser.add_argument('-o', '--out_path', type=str, help='output path', default='results/')
    parser.add_argument('--clip_len', type=int, default=243, help='clip length for network input')
    parser.add_argument('--subset', type=str, default='sp_MartialArtsStances_1', help='subset to use [train/test/special] (use sp_XXX_N to select only the first Nextracts containing XXX in their name)')
    opts = parser.parse_args()
    return opts

opts = parse_args()
args = get_config(opts.config)

args.num_new_joints = 38 if args.remap_joints_head else 0
model_backbone = load_backbone(args)

model_backbone = nn.DataParallel(model_backbone)
model_backbone = model_backbone.cuda()

print('Loading checkpoint', opts.evaluate)
checkpoint = torch.load(opts.evaluate, map_location=lambda storage, loc: storage)
model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
model_pos = model_backbone
model_pos.eval()
testloader_params = {
          'batch_size': 1,
          'shuffle': False,
          'num_workers': 1,
          'pin_memory': True,
          'prefetch_factor': 1,
          'persistent_workers': True,
          'drop_last': False
}

os.makedirs(opts.out_path, exist_ok=True)

dataset = EmbedRetargDataset(opts.data_path, 
                            clip_len=-1, # full length sequence
                            stride=-1,
                            root_rel_target=args.rootrel,
                            scale_by=args.scale_by,
                            scale_range=args.scale_range,
                            subset=opts.subset)


test_loader = DataLoader(dataset, **testloader_params)

embeddings_all = []
input_all = []
files_all = []
target_pos_all = []
seq_idx_all = []
with torch.no_grad():
    for batch_input, target_pos, file, seq_idx in tqdm(test_loader):
        input_all.append(batch_input)
        target_pos_all.append(target_pos)
        B, T = batch_input.shape[:2]
        assert(B == 1), "Only supported for batch size of 1"
        
        joints_num = 38 if args.remap_joints_head else 17
        embeddings = torch.zeros(B, T, joints_num, args.dim_rep//8)
        for i in range(0, T, args.maxlen):
            # make chunk by chunk inference with model maxlen
            # TODO : improve to make more efficient
            batch_input_chunk = batch_input[:, i:i+args.maxlen, :, :]
            batch_input_chunk = batch_input_chunk.cuda().float()
            embeddings_chunk = model_pos.module.get_reduced_features(batch_input_chunk)
            embeddings[:, i:i+args.maxlen, :, :] = embeddings_chunk
        
        embeddings_all.append(embeddings)
        files_all.append(file[0])
        seq_idx_all.append(seq_idx)


for file, embeddings_all in zip(files_all, embeddings_all):
    print(f"file : {file} shape of embeddings_all: {embeddings_all.shape}") # [B, T, J, dim_rep//8] i.e. [1, T, 17 or 38, 64]
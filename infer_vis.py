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
from tools.rendering import render_comparison

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
if torch.cuda.is_available():
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
                            clip_len=opts.clip_len,
                            stride=args.data_stride,
                            root_rel_target=args.rootrel,
                            scale_by=args.scale_by,
                            scale_range=args.scale_range,
                            subset=opts.subset)


test_loader = DataLoader(dataset, **testloader_params)

results_all = []
input_all = []
files_all = []
target_pos_all = []
seq_idx_all = []
with torch.no_grad():
    for batch_input, target_pos, file, seq_idx in tqdm(test_loader):
        input_all.append(batch_input)
        target_pos_all.append(target_pos)
        N, T = batch_input.shape[:2]
        if torch.cuda.is_available():
            batch_input = batch_input.cuda().float()
            
        # if args.no_conf:
        #     batch_input = batch_input[:, :, :, :2]
        # if args.flip:    
        #     batch_input_flip = flip_data(batch_input)
        #     predicted_3d_pos_1 = model_pos(batch_input)
        #     predicted_3d_pos_flip = model_pos(batch_input_flip)
        #     predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip) # Flip back
        #     predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
        # else:
        
        predicted_3d_pos = model_pos(batch_input)
        if args.rootrel:
            predicted_3d_pos[:,:,0,:]=0                    # [N,T,17,3]
        else:
            predicted_3d_pos[:,0,0,2]=0
            pass
        # if args.gt_2d:
        #     predicted_3d_pos[...,:2] = batch_input[...,:2]
        
        results_all.append(predicted_3d_pos) #.cpu().numpy())
        files_all.append(file[0])
        seq_idx_all.append(seq_idx)


for file, inp, result, target_pos, seq_idx in zip(files_all, input_all, results_all, target_pos_all, seq_idx_all):
    outfilename = file.replace(opts.data_path, '')
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    seq_idx_str = f'{seq_idx[0]:03d}'
    outfilename = outfilename.replace('.npz', f'{timestamp}_{seq_idx_str}.mp4')
    outfilename = outfilename.replace('/', '_')

    # data from the inference loop
    input_pos = inp[0].cpu().numpy()                           # (T, 17, 3)  x-right / y-down / conf
    output_pos = result[0].cpu().numpy()                       # (T, 17, 3)  right / down / front
    target_pos = target_pos[0].cpu().numpy()                   # (T, 38, 3)  right / front / up

    # data from the original file
    orig_data = np.load(file)
    start = seq_idx * 81
    end = start + opts.clip_len
    orig_pos = orig_data['body_pos_w'][start:end]          # (T, 22, 3)  right / front / up
    orig_quat = orig_data['body_quat_w'][start:end]      # (T, 22, 4)  x, y, z, w
    orig_quat = orig_quat[:, :, [1, 2, 3, 0]] # (T, 22, 4)  x, y, z, w
    
    # data from the G1 robot file
    g1_path = os.path.join(os.path.dirname(file), 'motion_shape_g1.npz')
    assert os.path.exists(g1_path), f"G1 data not found at {g1_path}"
    g1_data = np.load(g1_path)
    g1_pos = g1_data['body_pos_w'][start:end]          # (T, 38, 3)  right / front / up
    g1_quat = g1_data['body_quat_w'][start:end] if g1_pos is not None else None # (T, 38, 4)  w, x, y, z
    g1_quat = g1_quat[:, :, [1, 2, 3, 0]] # (T, 38, 4)  x, y, z, w
    print(g1_data["body_link_names"])

    # render the comparison video
    fps_in = 30
    save_path = os.path.join(opts.out_path, outfilename)
    render_comparison(orig_pos, input_pos, output_pos, target_pos, save_path,
                      fps=fps_in, g1_pos=g1_pos,
                      orig_quat=orig_quat, g1_quat=g1_quat)

    # save the data to a npz file
    output_npz = {
        'orig_pos': orig_pos,
        'input_pos': input_pos,
        'output_pos': output_pos,
    }
    if g1_pos is not None:
        output_npz['g1_pos'] = g1_pos
    np.savez(os.path.join(opts.out_path, outfilename.replace('.mp4', '.npz')), **output_npz)
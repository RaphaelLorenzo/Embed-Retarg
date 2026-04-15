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
from tools.rendering import load_original_pose_and_g1_data_from_file

timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pose3d/MB_ft_h36m_global_lite.yaml", help="Path to the config file.")
    parser.add_argument('-e', '--evaluate', default='checkpoints/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-d', '--data_path', type=str, help='data path', default='/home/raphael/Projects/github/accad_subset_random_shapes')
    parser.add_argument('-o', '--out_path', type=str, help='output path', default=f'results/')
    parser.add_argument('--clip_len', type=int, default=243, help='clip length for network input')
    parser.add_argument('--subset', type=str, default='sp_MartialArtsStances_1', help='subset to use [train/test/special] (use sp_XXX_N to select only the first Nextracts containing XXX in their name)')
    parser.add_argument('--save_video', action='store_true', help='save video (mp4 for visualization)')
    parser.add_argument('--save_pose3d', action='store_true', help='save pose data (npz for 3D pose visualization with vispy)')
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

model_dirname = opts.evaluate.split("/")[-2]
opts.out_path = os.path.join(opts.out_path, "model_"+model_dirname, "inference_"+timestamp)

os.makedirs(opts.out_path, exist_ok=True)


dataset = EmbedRetargDataset(opts.data_path, 
                            clip_len=-1, # full length sequence
                            stride=-1,
                            root_rel_target=args.rootrel,
                            scale_by=args.scale_by,
                            scale_range=args.scale_range,
                            subset=opts.subset,
                            test_keywords=args.test_keywords)


test_loader = DataLoader(dataset, **testloader_params)


with torch.no_grad():
    for batch_idx, (batch_input, target_pos, file, seq_idx) in enumerate(test_loader):
        # input_all.append(batch_input)
        # target_pos_all.append(target_pos)
        B, T = batch_input.shape[:2]
        assert(B == 1), "Only supported for batch size of 1"
        
        joints_num = 38 if args.remap_joints_head else 17
        features_type = "reduced" if args.use_compression else "full"
        features_dim = args.dim_rep//8 if args.use_compression else args.dim_rep
        embeddings = torch.zeros(B, T, joints_num, features_dim)
        outpose = torch.zeros(B, T, joints_num, 3)
        for i in range(0, T, args.maxlen):
            # make chunk by chunk inference with model maxlen
            # TODO : improve to make more efficient
            batch_input_chunk = batch_input[:, i:i+args.maxlen, :, :]
            batch_input_chunk = batch_input_chunk.cuda().float()
            
            embeddings_chunk = model_pos.module.get_features(batch_input_chunk, type=features_type)
            pose_chunk = model_pos(batch_input_chunk)
            
            embeddings[:, i:i+args.maxlen, :, :] = embeddings_chunk
            outpose[:, i:i+args.maxlen, :, :] = pose_chunk
        
        
        outfilename = file[0].replace(opts.data_path, '')
        seq_idx_str = f'{seq_idx[0]:03d}'
        outfilename = outfilename.replace('.npz', f'{seq_idx_str}')
        # outfilename = outfilename.replace('/', '_')
        
        save_embeddings_path = opts.out_path + outfilename + '_embed.npz'
        os.makedirs(os.path.dirname(save_embeddings_path), exist_ok=True)
        np.savez(save_embeddings_path, embeddings=embeddings.cpu().numpy())
        print(f"[{batch_idx}/{len(test_loader)}] Saved embeddings (shape {embeddings.shape}) to {save_embeddings_path}")

        if opts.save_video:
            # save the video
            orig_pos, orig_quat, g1_pos, g1_quat = load_original_pose_and_g1_data_from_file(file[0])
            save_video_path = opts.out_path + outfilename + '.mp4'
            os.makedirs(os.path.dirname(save_video_path), exist_ok=True)
            render_comparison(orig_pos, batch_input[0].cpu().numpy(), outpose[0].cpu().numpy(), target_pos[0].cpu().numpy(), save_video_path,
                            fps=30, g1_pos=g1_pos,
                            orig_quat=orig_quat, g1_quat=g1_quat)
            print(f"[{batch_idx}/{len(test_loader)}] Saved video (shape {outpose.shape}) to {save_video_path}")
        
        if opts.save_pose3d:
            # save the data to a npz file
            output_npz = {
                'orig_pos': orig_pos,
                'input_pos': batch_input[0].cpu().numpy(),
                'output_pos': outpose[0].cpu().numpy(),
                'g1_pos': g1_pos,
            }
            save_pose_npz_path = opts.out_path + outfilename + '_pose.npz'
            os.makedirs(os.path.dirname(save_pose_npz_path), exist_ok=True)
            np.savez(save_pose_npz_path, **output_npz)
            print(f"[{batch_idx}/{len(test_loader)}] Saved pose data dict (orig_pos, input_pos, output_pos, g1_pos) to {save_pose_npz_path}")

# -*- coding: utf-8 -*-
# @Author: Raphael
# @Date:   2024-10-09 11:02:29
# @Last Modified by:   Raphael
# @Last Modified time: 2024-10-14 15:26:52
import torch
import numpy as np
import ipdb
import glob
import os
import io
import math
import random
import json
import pickle
import math
from torch.utils.data import Dataset, DataLoader
from lib.utils.utils_data import crop_scale

def halpe2h36m(x):
    '''
        Input: x (T x V x C)  
       //Halpe 26 body keypoints
    {0,  "Nose"},
    {1,  "LEye"},
    {2,  "REye"},
    {3,  "LEar"},
    {4,  "REar"},
    {5,  "LShoulder"},
    {6,  "RShoulder"},
    {7,  "LElbow"},
    {8,  "RElbow"},
    {9,  "LWrist"},
    {10, "RWrist"},
    {11, "LHip"},
    {12, "RHip"},
    {13, "LKnee"},
    {14, "Rknee"},
    {15, "LAnkle"},
    {16, "RAnkle"},
    {17,  "Head"},
    {18,  "Neck"},
    {19,  "Hip"},
    {20, "LBigToe"},
    {21, "RBigToe"},
    {22, "LSmallToe"},
    {23, "RSmallToe"},
    {24, "LHeel"},
    {25, "RHeel"},
    '''
    T, V, C = x.shape
    y = np.zeros([T,17,C])
    y[:,0,:] = x[:,19,:]
    y[:,1,:] = x[:,12,:]
    y[:,2,:] = x[:,14,:]
    y[:,3,:] = x[:,16,:]
    y[:,4,:] = x[:,11,:]
    y[:,5,:] = x[:,13,:]
    y[:,6,:] = x[:,15,:]
    y[:,7,:] = (x[:,18,:] + x[:,19,:]) * 0.5
    y[:,8,:] = x[:,18,:]
    y[:,9,:] = x[:,0,:]
    y[:,10,:] = x[:,17,:]
    y[:,11,:] = x[:,5,:]
    y[:,12,:] = x[:,7,:]
    y[:,13,:] = x[:,9,:]
    y[:,14,:] = x[:,6,:]
    y[:,15,:] = x[:,8,:]
    y[:,16,:] = x[:,10,:]
    return y


def coco2h36m(x):
    '''
        Input: x (M x T x V x C)
        
        COCO: {0-nose 1-Leye 2-Reye 3-Lear 4Rear 5-Lsho 6-Rsho 7-Lelb 8-Relb 9-Lwri 10-Rwri 11-Lhip 12-Rhip 13-Lkne 14-Rkne 15-Lank 16-Rank}
        
        H36M:
        0: 'root',
        1: 'rhip',
        2: 'rkne',
        3: 'rank',
        4: 'lhip',
        5: 'lkne',
        6: 'lank',
        7: 'belly',
        8: 'neck',
        9: 'nose',
        10: 'head',
        11: 'lsho',
        12: 'lelb',
        13: 'lwri',
        14: 'rsho',
        15: 'relb',
        16: 'rwri'
    '''
    y = np.zeros(x.shape)
    y[:,0,:] = (x[:,11,:] + x[:,12,:]) * 0.5
    y[:,1,:] = x[:,12,:]
    y[:,2,:] = x[:,14,:]
    y[:,3,:] = x[:,16,:]
    y[:,4,:] = x[:,11,:]
    y[:,5,:] = x[:,13,:]
    y[:,6,:] = x[:,15,:]
    y[:,8,:] = (x[:,5,:] + x[:,6,:]) * 0.5
    y[:,7,:] = (y[:,0,:] + y[:,8,:]) * 0.5
    y[:,9,:] = x[:,0,:]
    y[:,10,:] = (x[:,1,:] + x[:,2,:]) * 0.5
    y[:,11,:] = x[:,5,:]
    y[:,12,:] = x[:,7,:]
    y[:,13,:] = x[:,9,:]
    y[:,14,:] = x[:,6,:]
    y[:,15,:] = x[:,8,:]
    y[:,16,:] = x[:,10,:]
    return y


def embedretarg2h36m(x):
    '''
        Input: x (T, 22, 3)
        
        0: 'pelvis'
        1: 'left_hip'
        2: 'right_hip'
        3: 'spine1'
        4: 'left_knee'
        5: 'right_knee'
        6: 'spine2' 
        7: 'left_ankle'
        8: 'right_ankle'
        9: 'spine3'
        10: 'left_foot'
        11: 'right_foot'
        12: 'neck'
        13: 'left_collar'
        14: 'right_collar'
        15: 'head'
        16: 'left_shoulder'
        17: 'right_shoulder'
        18: 'left_elbow'
        19: 'right_elbow'
        20: 'left_wrist'
        21: 'right_wrist'
        
        H36M:
        0: 'root',
        1: 'rhip',
        2: 'rkne',
        3: 'rank',
        4: 'lhip',
        5: 'lkne',
        6: 'lank',
        7: 'belly',
        8: 'neck',
        9: 'nose',
        10: 'head',
        11: 'lsho',
        12: 'lelb',
        13: 'lwri',
        14: 'rsho',
        15: 'relb',
        16: 'rwri'
    '''
    T, V, C = x.shape
    y = np.zeros([T, 17, C])
    # y[:, 0, :]  = x[:, 0, :]                          # root    ← pelvis
    y[:, 0, :]  = x[:, 2, :] * 0.5 + x[:, 1, :] * 0.5   # root    ← mid(left_hip, right_hip)
    y[:, 1, :]  = x[:, 2, :]                            # rhip    ← right_hip
    y[:, 2, :]  = x[:, 5, :]                            # rkne    ← right_knee
    y[:, 3, :]  = x[:, 8, :]                            # rank    ← right_ankle
    y[:, 4, :]  = x[:, 1, :]                            # lhip    ← left_hip
    y[:, 5, :]  = x[:, 4, :]                            # lkne    ← left_knee
    y[:, 6, :]  = x[:, 7, :]                            # lank    ← left_ankle
    y[:, 7, :]  = (x[:, 0, :] + x[:, 12, :]) * 0.5      # belly   ← mid(pelvis, neck)
    y[:, 8, :]  = x[:, 12, :]                           # neck    ← neck
    y[:, 9, :]  = (x[:, 12, :] + x[:, 15, :]) * 0.5     # nose    ← mid(neck, head)
    y[:, 10, :] = x[:, 15, :]                           # head    ← head
    y[:, 11, :] = x[:, 16, :]                           # lsho    ← left_shoulder
    y[:, 12, :] = x[:, 18, :]                           # lelb    ← left_elbow
    y[:, 13, :] = x[:, 20, :]                           # lwri    ← left_wrist
    y[:, 14, :] = x[:, 17, :]                           # rsho    ← right_shoulder
    y[:, 15, :] = x[:, 19, :]                           # relb    ← right_elbow
    y[:, 16, :] = x[:, 21, :]                           # rwri    ← right_wrist
    return y

    
def read_input(json_path, vid_size, scale_range, focus):
    with open(json_path, "r") as read_file:
        results = json.load(read_file)
    kpts_all = []
    image_ids = []
    kpts_3d_all = []
    for item in results:
        if focus!=None and item['idx']!=focus:
            continue
        kpts = np.array(item['keypoints']).reshape([-1,3])
        kpts_all.append(kpts)
        image_ids.append(item["image_id"])
        if "keypoints_3d" in item.keys():
            kpts_3d = np.array(item['keypoints_3d']).reshape([-1,3])
            kpts_3d_all.append(kpts_3d)        
        
    kpts_all = np.array(kpts_all)
    kpts_3d_all = np.array(kpts_3d_all)

    print(kpts_all.shape)

    if kpts_all.shape[1] == 26:
        kpts_all = halpe2h36m(kpts_all)
        if len(kpts_3d_all) > 0:
            assert(kpts_3d_all.shape[1] == 26)
            kpts_3d_all = halpe2h36m(kpts_3d_all)
            
    elif kpts_all.shape[1] == 17:
        print("WARNING : Using COCO17 input !")
        kpts_all = coco2h36m(kpts_all)
        if len(kpts_3d_all) > 0:
            assert(kpts_3d_all.shape[1] == 17)
            kpts_3d_all = coco2h36m(kpts_3d_all)
    else:
        print("Error, expecting kpts_all of shape [..., 17 or 26, ...]")
        exit(0)

    
    if vid_size:
        w, h = vid_size
        scale = min(w,h) / 2.0
        kpts_all[:,:,:2] = kpts_all[:,:,:2] - np.array([w, h]) / 2.0
        kpts_all[:,:,:2] = kpts_all[:,:,:2] / scale
        motion = kpts_all
        
    if scale_range:
        motion = crop_scale(kpts_all, scale_range) 
    
    motion_3d = kpts_3d_all.astype(np.float32)
        
    return motion.astype(np.float32), image_ids, motion_3d

class WildDetDataset(Dataset):
    def __init__(self, json_path, clip_len=243, vid_size=None, scale_range=None, focus=None):
        self.json_path = json_path
        self.clip_len = clip_len
        self.vid_all, self.image_ids, self.motion_3d = read_input(json_path, vid_size, scale_range, focus)
        
    def __len__(self):
        'Denotes the total number of samples'
        return math.ceil(len(self.vid_all) / self.clip_len)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        st = index*self.clip_len
        end = min((index+1)*self.clip_len, len(self.vid_all))
        return self.vid_all[st:end]


H36M_INTRINSICS = {
    "K":
    [
        [
            1145.04940458804,
            0.0,
            512.541504956548
        ],
        [
            0.0,
            1143.78109572365,
            515.4514869776
        ],
        [
            0.0,
            0.0,
            1.0
        ]
    ],
    "distortion": 
    [
        -0.207098910824901,
        0.247775183068982,
        -0.00142447157470321,
        -0.000975698859470499,
        -0.00307515035078854
    ]
}



def project_to_image(positions_3d, cam_position, cam_rotation, intrinsics):
    """Project 3D world positions onto a virtual image plane.

    Args:
        positions_3d: (T, V, 3) world coordinates (right, front, up)
        cam_position: (3,) camera location in world space
        cam_rotation: (3,) extra Euler angles (rx, ry, rz) in radians applied
                      on top of the base world→camera rotation
        intrinsics:   dict with ``"K"`` (3×3 list) and optional ``"distortion"``

    Returns:
        (T, V, 3) with channels (u, v, confidence=1.0)
    """
    T, V, _ = positions_3d.shape
    K = np.array(intrinsics["K"], dtype=np.float64)

    # Base rotation: world (right, front, up) → camera (right, down, forward)
    R_base = np.array([[1,  0,  0],
                       [0,  0, -1],
                       [0,  1,  0]], dtype=np.float64)

    # Extra rotation from Euler angles (XYZ intrinsic order)
    rx, ry, rz = cam_rotation
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    R = Rz @ Ry @ Rx @ R_base

    t = np.asarray(cam_position, dtype=np.float64)

    # World → camera: translate then rotate  (T, V, 3)
    pts_cam = (positions_3d.astype(np.float64) - t) @ R.T

    # Perspective divide
    z = np.clip(pts_cam[:, :, 2:3], 1e-6, None)
    xy = pts_cam[:, :, :2] / z

    # Intrinsics → pixel coords
    result = np.empty((T, V, 3), dtype=np.float32)
    result[:, :, 0] = (K[0, 0] * xy[:, :, 0] + K[0, 2]).astype(np.float32)
    result[:, :, 1] = (K[1, 1] * xy[:, :, 1] + K[1, 2]).astype(np.float32)
    result[:, :, 2] = 1.0
    return result


class EmbedRetargDataset(Dataset):
    def __init__(self, data_path, max_len=243, scale_range=None, project_to_image_params={"cam_position":(0.0,-5.0,1.5), "cam_rotation":(0.0,0.0,0.0), "intrinsics":H36M_INTRINSICS}):
        self.max_len = max_len
        self.scale_range = scale_range
        self.project_to_image_params = project_to_image_params # do proper reprojection or just use x,z coordinates
        
        # List all npz files in the data path
        files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_path)) for f in fn if f.endswith(".npz")]
        files.sort()
        
        invalid_filenames = ["motion_shape_g1.npz", "motion_shape.npz"]
        files = [file for file in files if not any(invalid_filename in file for invalid_filename in invalid_filenames)]
        
        self.files = files
        print(f"Found {len(self.files)} files")
        print(self.files)
        
        self.files = [file for file in files if "Walking" in file]
        self.files = self.files[:1]
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.files)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        file = self.files[index]
        data = np.load(file)
        
        # print(data.keys()) 
        # KeysView(NpzFile 
        # '/home/raphael/Projects/github/accad_subset_random_shapes/Female1General_c3d/A1_-_Stand_stageii/motion_shape.npz' 
        # with keys: body_link_names, body_pos_w, body_quat_w, betas, fps)
        
        # print(data['body_link_names']) 
        # ['pelvis' 'left_hip' 'right_hip' 'spine1' 'left_knee' 'right_knee'
        # 'spine2' 'left_ankle' 'right_ankle' 'spine3' 'left_foot' 'right_foot'
        # 'neck' 'left_collar' 'right_collar' 'head' 'left_shoulder'
        # 'right_shoulder' 'left_elbow' 'right_elbow' 'left_wrist' 'right_wrist']
        
        print(data['body_pos_w'].shape) # (T, 22, 3) (xyz -> right, front, up)
        
        if self.project_to_image_params is not None:
            positions = project_to_image(
                data['body_pos_w'],
                self.project_to_image_params['cam_position'],
                self.project_to_image_params['cam_rotation'],
                self.project_to_image_params['intrinsics'])
            
            # ### DEBUG MATPLOTLIB ###
            # print(positions.shape)
            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(10, 10))
            # plt.scatter(positions[0, :, 0], positions[0, :, 1])
            # plt.savefig('debug_positions.png')
            # exit(0)
            #########################################################
            
            
        else:
            positions = data['body_pos_w']
            positions = positions[:, :, [0, 2, 1]] # (T, 22, 3) (x, z, y)
            positions[:, :, 1] = -positions[:, :, 1] # (T, 22, 3) (x, -z, y)
            positions[:, :, 2] = 1.0 # full confidence
            
            
        position_h36m_2d = embedretarg2h36m(positions)
        
        if position_h36m_2d.shape[0] > self.max_len:
            position_h36m_2d = position_h36m_2d[:self.max_len]

        if self.scale_range:
            position_h36m_2d = crop_scale(position_h36m_2d, self.scale_range) 
    
        return position_h36m_2d, file
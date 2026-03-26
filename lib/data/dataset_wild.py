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
from lib.utils.utils_data import crop_scale, crop_scale_frame

from scipy.spatial.transform import Rotation as R_scipy

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


def g12h36m(x):
    '''
        Input: x (T, 38, 3)
        
        G1:
        0: 'pelvis'
        1: 'left_hip_pitch_link'
        2: 'left_hip_roll_link'
        3: 'left_hip_yaw_link'
        4: 'left_knee_link'
        5: 'left_ankle_pitch_link'
        6: 'left_ankle_roll_link'
        7: 'left_toe_link'
        8: 'pelvis_contour_link'
        9: 'right_hip_pitch_link'
        10: 'right_hip_roll_link'
        11: 'right_hip_yaw_link'
        12: 'right_knee_link'
        13: 'right_ankle_pitch_link'
        14: 'right_ankle_roll_link'
        15: 'right_toe_link'
        16: 'waist_yaw_link'
        17: 'waist_roll_link'
        18: 'torso_link'
        19: 'head_link'
        20: 'head_mocap'
        21: 'imu_in_torso'
        22: 'left_shoulder_pitch_link'
        23: 'left_shoulder_roll_link'
        24: 'left_shoulder_yaw_link'
        25: 'left_elbow_link'
        26: 'left_wrist_roll_link'
        27: 'left_wrist_pitch_link'
        28: 'left_wrist_yaw_link'
        29: 'left_rubber_hand'
        30: 'right_shoulder_pitch_link'
        31: 'right_shoulder_roll_link'
        32: 'right_shoulder_yaw_link'
        33: 'right_elbow_link'
        34: 'right_wrist_roll_link'
        35: 'right_wrist_pitch_link'
        36: 'right_wrist_yaw_link'
        37: 'right_rubber_hand'
 
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
    y[:, 0, :]  = x[:, 0, :]                              # root  ← pelvis
    y[:, 1, :]  = x[:, 9, :]                              # rhip  ← right_hip_pitch_link
    y[:, 2, :]  = x[:, 12, :]                             # rkne  ← right_knee_link
    y[:, 3, :]  = x[:, 13, :]                             # rank  ← right_ankle_pitch_link
    y[:, 4, :]  = x[:, 1, :]                              # lhip  ← left_hip_pitch_link
    y[:, 5, :]  = x[:, 4, :]                              # lkne  ← left_knee_link
    y[:, 6, :]  = x[:, 5, :]                              # lank  ← left_ankle_pitch_link
    y[:, 7, :]  = (x[:, 0, :] + x[:, 18, :]) * 0.5       # belly ← mid(pelvis, torso_link)
    y[:, 8, :]  = x[:, 18, :]                             # neck  ← torso_link
    y[:, 9, :]  = (x[:, 18, :] + x[:, 19, :]) * 0.5      # nose  ← mid(torso_link, head_link)
    y[:, 10, :] = x[:, 19, :]                             # head  ← head_link
    y[:, 11, :] = x[:, 22, :]                             # lsho  ← left_shoulder_pitch_link
    y[:, 12, :] = x[:, 25, :]                             # lelb  ← left_elbow_link
    y[:, 13, :] = x[:, 28, :]                             # lwri  ← left_wrist_yaw_link
    y[:, 14, :] = x[:, 30, :]                             # rsho  ← right_shoulder_pitch_link
    y[:, 15, :] = x[:, 33, :]                             # relb  ← right_elbow_link
    y[:, 16, :] = x[:, 36, :]                             # rwri  ← right_wrist_yaw_link
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


def root_align(positions, quaternions):
    """Re-express joint positions in the root frame each frame.

    For every frame the root (joint 0) is moved to the origin and all
    joints are rotated by the inverse of the root orientation, so the
    skeleton faces a fixed direction with zero root yaw/pitch/roll.

    Args:
        positions:   (T, V, 3) world joint positions
        quaternions: (T, V, 4) world joint orientations (x, y, z, w)

    Returns:
        aligned: (T, V, 3) root-relative, root-orientation-aligned positions
    """
    T, V, _ = positions.shape
    root_pos  = positions[:, 0, :]      # (T, 3)
    root_quat = quaternions[:, 0, :]    # (T, 4)  xyzw

    root_rot_inv = R_scipy.from_quat(root_quat).inv()

    translated = positions - root_pos[:, np.newaxis, :]          # (T, V, 3)
    aligned = np.empty_like(translated)
    for t in range(T):
        aligned[t] = root_rot_inv[t].apply(translated[t])
    return aligned

class EmbedRetargDataset(Dataset):
    def __init__(self, 
                 data_path, 
                 max_len=243, 
                 stride=81, 
                 root_rel_target=True,
                 scale_by="sequence",
                 scale_range=[1,1], 
                 project_to_image_params={"cam_position":(0.0,-5.0,1.5), "cam_rotation":(0.0,0.0,0.0), "intrinsics":H36M_INTRINSICS},
                 subset="train",
                 test_keywords=[]):
        
        self.subset = subset
        self.max_len = max_len
        self.stride = stride
        self.scale_range = scale_range
        self.project_to_image_params = project_to_image_params # do proper reprojection or just use x,z coordinates
        self.root_rel_target = root_rel_target
        self.scale_by = scale_by
        
        # List all npz files in the data path
        files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_path)) for f in fn if f.endswith(".npz")]
        files.sort()
        self.files = files
        print(f"Subset {subset} : Found {len(self.files)} files")            
        
        if subset == "special_walking":
            invalid_filenames = ["motion_shape_g1.npz", "random_shape_"]
            self.files = [file for file in self.files if not any(invalid_filename in file for invalid_filename in invalid_filenames)]
            self.files = [file for file in self.files if "Walking" in file]
            self.files = self.files[:1]
            
        elif subset == "special_stances":
            invalid_filenames = ["motion_shape_g1.npz", "random_shape_"]
            self.files = [file for file in self.files if not any(invalid_filename in file for invalid_filename in invalid_filenames)]
            self.files = [file for file in self.files if "MartialArtsStances_c3d" in file]
            self.files = self.files[:1]
            
        elif subset == "train" and len(test_keywords) == 0:
            invalid_filenames = ["motion_shape_g1.npz", "motion_shape.npz"]
            self.files = [file for file in self.files if not any(invalid_filename in file for invalid_filename in invalid_filenames)]
        elif subset == "test" and len(test_keywords) == 0:
            invalid_filenames = ["motion_shape_g1.npz", "random_shape_"]
            self.files = [file for file in self.files if not any(invalid_filename in file for invalid_filename in invalid_filenames)]
        
        elif subset == "train" and len(test_keywords) > 0:
            invalid_filenames = ["motion_shape_g1.npz"] + test_keywords
            self.files = [file for file in self.files if not any(invalid_filename in file for invalid_filename in invalid_filenames)]
            
        elif subset == "test" and len(test_keywords) > 0:
            invalid_filenames = ["motion_shape_g1.npz"]
            self.files = [file for file in self.files if not any(invalid_filename in file for invalid_filename in invalid_filenames)]
            self.files = [file for file in self.files if any(test_keyword in file for test_keyword in test_keywords)]
                        
        else:
            raise ValueError(f"Invalid subset: {subset} and test_keywords: {test_keywords}")
        
        print(f"Subset {subset} : Kept {len(self.files)} files")            
        
        self.inputs = [] # list of (file, seq_idx, start, end) to mitigate the long sequences
        rejected_inputs = []
        for file in self.files:
            data = np.load(file)
            body_pos_w = data['body_pos_w']
            seq_len = body_pos_w.shape[0]
            
            if seq_len > self.max_len:
                for seq_idx,i in enumerate(range(0, seq_len, self.stride)):
                    if i + self.max_len > seq_len:
                        # too short sequence, skip
                        rejected_inputs.append((file, seq_idx, i, i + self.max_len, "too short subseq"))
                        continue
                    
                    self.inputs.append((file, seq_idx, i, i + self.max_len))

                    # check we can be visible in the reprojection (approximatively)
                    if self.project_to_image_params is not None:
                        cam_position = self.project_to_image_params['cam_position']
                        cam_x = cam_position[0]
                        cam_y = cam_position[1]
                        
                        if not np.all(body_pos_w[:, :, 1] > cam_y+1.0):
                            rejected_inputs.append((file, seq_idx, i, i + self.max_len, "not in front of the camera"))
                            continue
                        
                        x_rel = body_pos_w[:, :, 0] - cam_x
                        y_rel = body_pos_w[:, :, 1] - cam_y
                        angles = np.arctan(x_rel / y_rel)
                        angles = np.rad2deg(angles)
                        
                        # print(x_rel[0], y_rel[0], angles[0])
                        
                        if not np.all(angles < 30.0):
                            rejected_inputs.append((file, seq_idx, i, i + self.max_len, "too far on the left or right of the camera"))
                            continue
                        if not np.all(angles > -30.0):
                            rejected_inputs.append((file, seq_idx, i, i + self.max_len, "too far on the left or right of the camera"))
                            continue
                    
            elif seq_len == self.max_len:
                self.inputs.append((file, 0, 0, seq_len))
                
            else:
                # too short sequence, skip
                rejected_inputs.append((file, 0, 0, seq_len, "too short seq"))
                continue
        
        print(f"Subset {subset} : Accepted {len(self.inputs)} inputs")
        print(f"Subset {subset} : Rejected {len(rejected_inputs)} inputs")
        # print(rejected_inputs)
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        file, seq_idx, start, end = self.inputs[index]
        data = np.load(file)
        
        input_body_pos_w = data['body_pos_w'][start:end]
        # input_body_quat_w = data['body_quat_w'][start:end]
        
        target_file = os.path.join(os.path.dirname(file), 'motion_shape_g1.npz')
        target_data = np.load(target_file)
        target_body_pos_w = target_data['body_pos_w'][start:end]          # (T, 38, 3)  right / front / up
        target_body_quat_w = target_data['body_quat_w'][start:end]      # (T, 38, 4)  w, x, y, z
        
        if self.root_rel_target:
            target_body_quat_w = target_body_quat_w[:, :, [1, 2, 3, 0]] # (T, 38, 4)  x, y, z, w
            target_pos = root_align(target_body_pos_w, target_body_quat_w) # (T, 38, 3)  right / front / up

        if self.project_to_image_params is not None:
            positions = project_to_image(
                input_body_pos_w,
                self.project_to_image_params['cam_position'],
                self.project_to_image_params['cam_rotation'],
                self.project_to_image_params['intrinsics'])
        else:
            positions = input_body_pos_w
            positions = positions[:, :, [0, 2, 1]] # (T, 22, 3) (x, z, y)
            positions[:, :, 1] = -positions[:, :, 1] # (T, 22, 3) (x, -z, y)
            positions[:, :, 2] = 1.0 # full confidence
            
            
        position_h36m_2d = embedretarg2h36m(positions)
        
        if self.scale_by == "sequence":
            position_h36m_2d = crop_scale(position_h36m_2d, self.scale_range) 
        elif self.scale_by == "frame":
            position_h36m_2d = crop_scale_frame(position_h36m_2d, self.scale_range)
    
        return position_h36m_2d, target_pos, file, seq_idx

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
from tools.conversion_tools import embedretarg2h36m

from scipy.spatial.transform import Rotation as R_scipy

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

class EmbedRetargDataset(Dataset):
    def __init__(self, 
                 data_path, 
                 clip_len=243, 
                 stride=81, 
                 root_rel_target=True,
                 scale_by="sequence",
                 scale_range=[1,1], 
                 project_to_image_params={"cam_position":(0.0,-5.0,1.5), "cam_rotation":(0.0,0.0,0.0), "intrinsics":H36M_INTRINSICS},
                 subset="train",
                 test_keywords=[]):
        
        self.subset = subset
        self.clip_len = clip_len
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
        
        if subset.startswith("sp_"):
            invalid_filenames = ["motion_shape_g1.npz", "random_shape_"]
            sp_name_split = subset.split("_")
            assert len(sp_name_split) == 3, "Special subset must be of the form sp_XXX_N"
            sp_name = sp_name_split[1]
            sp_number = int(sp_name_split[2])
            self.files = [file for file in self.files if not any(invalid_filename in file for invalid_filename in invalid_filenames)]
            self.files = [file for file in self.files if sp_name in file]            
            sp_number = min(sp_number, len(self.files))
            print(f"Subset {sp_name} : Kept {sp_number} files (requested {subset})")
            self.files = self.files[:sp_number]
            
        # case of train/test split by using motion_shape_XXX.npz for tests and random_shape_XXX.npz for train
        elif subset == "train" and len(test_keywords) == 0:
            invalid_filenames = ["motion_shape_g1.npz", "motion_shape.npz"]
            self.files = [file for file in self.files if not any(invalid_filename in file for invalid_filename in invalid_filenames)]
        elif subset == "test" and len(test_keywords) == 0:
            invalid_filenames = ["motion_shape_g1.npz", "random_shape_"]
            self.files = [file for file in self.files if not any(invalid_filename in file for invalid_filename in invalid_filenames)]
        
        # case of train/test split by using test_keywords for tests
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
            
            if seq_len > self.clip_len:
                for seq_idx,i in enumerate(range(0, seq_len, self.stride)):
                    if i + self.clip_len > seq_len:
                        # too short sequence, skip
                        rejected_inputs.append((file, seq_idx, i, i + self.clip_len, "too short subseq"))
                        continue
                    
                    self.inputs.append((file, seq_idx, i, i + self.clip_len))

                    # check we can be visible in the reprojection (approximatively)
                    if self.project_to_image_params is not None:
                        cam_position = self.project_to_image_params['cam_position']
                        cam_x = cam_position[0]
                        cam_y = cam_position[1]
                        
                        if not np.all(body_pos_w[:, :, 1] > cam_y+1.0):
                            rejected_inputs.append((file, seq_idx, i, i + self.clip_len, "not in front of the camera"))
                            continue
                        
                        x_rel = body_pos_w[:, :, 0] - cam_x
                        y_rel = body_pos_w[:, :, 1] - cam_y
                        angles = np.arctan(x_rel / y_rel)
                        angles = np.rad2deg(angles)
                        
                        # print(x_rel[0], y_rel[0], angles[0])
                        
                        if not np.all(angles < 30.0):
                            rejected_inputs.append((file, seq_idx, i, i + self.clip_len, "too far on the left or right of the camera"))
                            continue
                        if not np.all(angles > -30.0):
                            rejected_inputs.append((file, seq_idx, i, i + self.clip_len, "too far on the left or right of the camera"))
                            continue
                    
            elif seq_len == self.clip_len:
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
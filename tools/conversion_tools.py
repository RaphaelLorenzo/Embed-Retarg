import numpy as np
import torch

def g12h36m_torch(x):
    '''
        Input: x (B, T, 38, 3)
        
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
    
    B, T, V, C = x.shape
    y = torch.zeros([B, T, 17, C], device=x.device, dtype=x.dtype)
    y[:, :, 0, :]  = x[:, :, 0, :]                              # root  ← pelvis
    y[:, :, 1, :]  = x[:, :, 9, :]                              # rhip  ← right_hip_pitch_link
    y[:, :, 2, :]  = x[:, :, 12, :]                             # rkne  ← right_knee_link
    y[:, :, 3, :]  = x[:, :, 13, :]                             # rank  ← right_ankle_pitch_link
    y[:, :, 4, :]  = x[:, :, 1, :]                              # lhip  ← left_hip_pitch_link
    y[:, :, 5, :]  = x[:, :, 4, :]                              # lkne  ← left_knee_link
    y[:, :, 6, :]  = x[:, :, 5, :]                              # lank  ← left_ankle_pitch_link
    y[:, :, 7, :]  = (x[:, :, 0, :] + x[:, :, 18, :]) * 0.5     # belly ← mid(pelvis, torso_link)
    y[:, :, 8, :]  = x[:, :, 18, :]                             # neck  ← torso_link
    y[:, :, 9, :]  = (x[:, :, 18, :] + x[:, :, 19, :]) * 0.5    # nose  ← mid(torso_link, head_link)
    y[:, :, 10, :] = x[:, :, 19, :]                             # head  ← head_link
    y[:, :, 11, :] = x[:, :, 22, :]                             # lsho  ← left_shoulder_pitch_link
    y[:, :, 12, :] = x[:, :, 25, :]                             # lelb  ← left_elbow_link
    y[:, :, 13, :] = x[:, :, 28, :]                             # lwri  ← left_wrist_yaw_link
    y[:, :, 14, :] = x[:, :, 30, :]                             # rsho  ← right_shoulder_pitch_link
    y[:, :, 15, :] = x[:, :, 33, :]                             # relb  ← right_elbow_link
    y[:, :, 16, :] = x[:, :, 36, :]                             # rwri  ← right_wrist_yaw_link
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


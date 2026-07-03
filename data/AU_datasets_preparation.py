import numpy as np
import pandas as pd
import os
from config.params import *
from config.datasets import *

# Get the data of the participant sub_ID in the DISFA dataset
def DISFA_get_subject(sub_ID, camera = 'left', label_type = "intensity"):
    if camera == 'left': # Left-view
        DISFA_img_rootdir = DISFA_IMG_ROOTDIR_LEFTCAM
    elif camera == 'right': # Right-view
        DISFA_img_rootdir = DISFA_IMG_ROOTDIR_RIGHTCAM
    else:
        raise ValueError('Invalid Camera Direction')

    img_dir = os.path.join(f'{DISFA_img_rootdir}_preprocessed', sub_ID)

    img_names = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    img_names = sorted(img_names, key=lambda x: int(x.split('.')[0]))
    img_paths = [os.path.join(img_dir, name) for name in img_names]
        
    frame_indices = [int(name.split('.')[0]) for name in img_names]
    df_AUs = []
    for AU in DISFA_AUS_IE:
        label_path = os.path.join(DISFA_LABEL_ROOTDIR, sub_ID, f"{sub_ID}_au{AU}.txt")
        df_AU = pd.read_csv(label_path, header=None, names=[AU], index_col=0)
        df_AUs.append(df_AU)
    df_AUs = pd.concat(df_AUs, axis=1)

    labels = []
    for frame_index in frame_indices:
        if label_type == "intensity": # AU intensity estimation
            frame_labels = -np.ones(len(DISFA_AUS_IE)) * INVALID_INDICATOR_IE
            for i, AU in enumerate(DISFA_AUS_IE):
                if frame_index in df_AUs[AU].index:
                    label = df_AUs[AU][frame_index]
                    if 0 <= label <= 5:
                        frame_labels[i] = label
        elif label_type == "binary occurrence": # AU detection
            frame_labels = -np.ones(len(DISFA_AUS_D)) * INVALID_INDICATOR_D
            for i, AU in enumerate(DISFA_AUS_D):
                if frame_index in df_AUs[AU].index:
                    label = df_AUs[AU][frame_index]
                    if 0 <= label <= 5:
                        frame_labels[i] = int(label >= BINARY_OCCURRENCE_THRESHOLD)
        labels.append(frame_labels)
    return img_paths, labels

# Get the data of the participant sub_ID in the DISFA+ dataset
def DISFAPlus_get_subject(sub_ID, label_type = "intensity"):
    all_img_paths = []
    all_labels = []
    img_rootdir = os.path.join(f'{DISFAPLUS_IMG_ROOTDIR}_preprocessed', sub_ID)
    trials = os.listdir(img_rootdir)
    trials = sorted(trials)
    for trial in trials:
        img_dir = os.path.join(img_rootdir, trial)
        img_names = os.listdir(img_dir)
        img_names = sorted(img_names, key=lambda x: int(x.split('.')[0]))
        img_paths = [os.path.join(img_dir, name) for name in img_names]
        all_img_paths.extend(img_paths)
        
        df_AUs = []
        for AU in DISFAPLUS_AUS:
            label_path = os.path.join(DISFAPLUS_LABEL_ROOTDIR, sub_ID, trial, f'AU{AU}.txt')
            df_AU = pd.read_csv(label_path, header=None, delimiter='     ', skiprows=2, names=[AU], index_col=0, engine='python')
            df_AUs.append(df_AU)
        df_AUs = pd.concat(df_AUs, axis=1)
        
        labels = []
        for img_name in img_names:
            if label_type == "intensity": # AU intensity estimation
                frame_labels = -np.ones(len(DISFAPLUS_AUS)) * INVALID_INDICATOR_IE
                for i, AU in enumerate(DISFAPLUS_AUS):
                    label = df_AUs[AU][img_name]
                    if 0 <= label <= 5:
                        frame_labels[i] = label
            elif label_type == "binary occurrence": # AU detection
                frame_labels = -np.ones(len(DISFAPLUS_AUS)) * INVALID_INDICATOR_D
                for i, AU in enumerate(DISFAPLUS_AUS):
                    label = df_AUs[AU][img_name]
                    if 0 <= label <= 5:
                        frame_labels[i] = int(label >= BINARY_OCCURRENCE_THRESHOLD)
            labels.append(frame_labels)
        all_labels.extend(labels)
    return all_img_paths, all_labels

# Get the data of the participant sub_ID in the UNBC-McMaster dataset
def UNBCMcMaster_get_subject(sub_ID, label_type = "intensity"):
    all_img_paths = []
    all_labels = []
    img_rootdir = os.path.join(f'{UNBCMCMASTER_IMG_ROOTDIR}_preprocessed', sub_ID)
    tests = os.listdir(img_rootdir)
    tests = sorted(tests)
    for test in tests:
        img_dir = os.path.join(img_rootdir, test)
        img_names = os.listdir(img_dir)
        img_names = sorted(img_names, key=lambda x: int(x.split('.')[0][-3:]))
        img_paths = [os.path.join(img_dir, name) for name in img_names]
        all_img_paths.extend(img_paths)
        
        frame_indices = [int(name.split('.')[0][-3:]) for name in img_names]
        labels = []
        for frame_index in frame_indices:
            label_path = os.path.join(UNBCMCMASTER_LABEL_ROOTDIR, sub_ID, test, test + '{:03d}'.format(frame_index) + '_facs.txt')
            df_AU = pd.read_csv(label_path, sep='   ', header=None, names = ['AU', 'intensity', 'onset', 'offset'], engine = 'python', dtype=int)
            
            if label_type == "intensity": # AU intensity estimation
                frame_labels = np.zeros(len(UNBCMCMASTER_AUS_IE))
                for i, row in df_AU.iterrows():
                    AU = row['AU']
                    label = row['intensity']
                    if AU in UNBCMCMASTER_AUS_IE:
                        if 0 <= label <= 5:
                            frame_labels[UNBCMCMASTER_AUS_IE.index(AU)] = label
                        else:
                            frame_labels[UNBCMCMASTER_AUS_IE.index(AU)] = -INVALID_INDICATOR_IE
            elif label_type == "binary occurrence": # AU detection
                frame_labels = np.zeros(len(UNBCMCMASTER_AUS_D))
                for i, row in df_AU.iterrows():
                    AU = row['AU']
                    label = row['intensity']
                    if AU in UNBCMCMASTER_AUS_D:
                        if 0 <= label <= 5:
                            if AU != AU_EYE_CLOSURE:
                                frame_labels[UNBCMCMASTER_AUS_D.index(AU)] = int(label >= BINARY_OCCURRENCE_THRESHOLD)
                            else:
                                frame_labels[UNBCMCMASTER_AUS_D.index(AU)] = int(label >= 1)
                        else:
                            frame_labels[UNBCMCMASTER_AUS_D.index(AU)] = -INVALID_INDICATOR_D
            labels.append(frame_labels)
        all_labels.extend(labels)
    return all_img_paths, all_labels
import numpy as np
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import warnings
from config.datasets import *
from data.AU_datasets_preparation import *

# Extracting individual frames in AU datasets
class AU_dataset_single_frames(Dataset):
    def __init__(self, subject_dict, transform=None, label_type="intensity"):
        all_img_paths = []
        all_labels = []
        if 'DISFAleft' in subject_dict: # Left-view videos of the DISFA dataset
            sub_IDs = subject_dict['DISFAleft']
            for sub_ID in sub_IDs:
                img_paths, labels = DISFA_get_subject(sub_ID, camera = 'left', label_type=label_type)
                all_img_paths.extend(img_paths)
                all_labels.extend(labels)
        if 'DISFAright' in subject_dict: # Right-view videos of the DISFA dataset
            sub_IDs = subject_dict['DISFAright']
            for sub_ID in sub_IDs:
                img_paths, labels = DISFA_get_subject(sub_ID, camera = 'right', label_type=label_type)
                all_img_paths.extend(img_paths)
                all_labels.extend(labels)
        if 'DISFAPlus' in subject_dict: # DISFA+ dataset
            sub_IDs = subject_dict['DISFAPlus']
            for sub_ID in sub_IDs:
                img_paths, labels = DISFAPlus_get_subject(sub_ID, label_type=label_type)
                all_img_paths.extend(img_paths)
                all_labels.extend(labels)
        if 'UNBCMcMaster' in subject_dict: #UNBC-McMaster dataset
            sub_IDs = subject_dict['UNBCMcMaster']
            for sub_ID in sub_IDs:
                img_paths, labels = UNBCMcMaster_get_subject(sub_ID, label_type=label_type)
                all_img_paths.extend(img_paths)
                all_labels.extend(labels)
        
        self.all_img_paths = all_img_paths
        self.all_labels = all_labels
        self.transform = transform

    def __len__(self):
        return len(self.all_img_paths)

    def __getitem__(self, index):
        path = self.all_img_paths[index]
        img = cv2.imread(path)
        label = self.all_labels[index]
        if self.transform:
            img = self.transform(img)
        warnings.filterwarnings("ignore", category=UserWarning)
        return path, torch.tensor(img, dtype=torch.float), torch.tensor(label, dtype=torch.float)


# Extracting frame pairs (the reference frame and the target frame) in AU datasets
class AU_dataset_frame_pairs(Dataset):
    def __init__(self, subject_dict, transform=None, label_type="intensity"):
        all_baseline_img_paths = []
        all_img_paths = []
        all_labels = []
        if 'DISFAleft' in subject_dict: # Left-view videos of the DISFA dataset
            sub_IDs = subject_dict['DISFAleft']
            for sub_ID in sub_IDs:
                img_paths, labels = DISFA_get_subject(sub_ID, camera = 'left', label_type=label_type)
                all_baseline_img_paths.extend([img_paths[DISFA_BASELINE_FRAMES[sub_ID]-1]] * len(img_paths))
                all_img_paths.extend(img_paths)
                all_labels.extend(labels)

        if 'DISFAright' in subject_dict: # Right-view videos of the DISFA dataset
            sub_IDs = subject_dict['DISFAright']
            for sub_ID in sub_IDs:
                img_paths, labels = DISFA_get_subject(sub_ID, camera = 'right', label_type=label_type)
                all_baseline_img_paths.extend([img_paths[DISFA_BASELINE_FRAMES[sub_ID]-1]] * len(img_paths))
                all_img_paths.extend(img_paths)
                all_labels.extend(labels)

        if 'DISFAPlus' in subject_dict: # DISFA+ dataset
            sub_IDs = subject_dict['DISFAPlus']
            for sub_ID in sub_IDs:
                img_paths, labels = DISFAPlus_get_subject(sub_ID, label_type=label_type)
                all_baseline_img_paths.extend([img_paths[[DISFAPLUS_BASELINE_FRAMES[sub_ID] in path for path in img_paths].index(True)]] * len(img_paths))
                all_img_paths.extend(img_paths)
                all_labels.extend(labels)

        if 'UNBCMcMaster' in subject_dict: #UNBC-McMaster dataset
            sub_IDs = subject_dict['UNBCMcMaster']
            for sub_ID in sub_IDs:
                img_paths, labels = UNBCMcMaster_get_subject(sub_ID, label_type=label_type)
                all_baseline_img_paths.extend([img_paths[[UNBCMCMASTER_BASELINE_FRAMES[sub_ID] in path for path in img_paths].index(True)]] * len(img_paths))
                all_img_paths.extend(img_paths)
                all_labels.extend(labels)
        
        self.all_baseline_img_paths = all_baseline_img_paths
        self.all_img_paths = all_img_paths
        self.all_labels = all_labels
        self.transform = transform

    def __len__(self):
        return len(self.all_img_paths)

    def __getitem__(self, index):
        baseline_img_path = self.all_baseline_img_paths[index]
        img_path = self.all_img_paths[index]
        baseline_img = cv2.imread(baseline_img_path)
        img = cv2.imread(img_path)
        label = self.all_labels[index]
        if self.transform:
            baseline_img = self.transform(baseline_img)
            img = self.transform(img)
        
        warnings.filterwarnings("ignore", category=UserWarning)
        img_path_pair = ' '.join([baseline_img_path, img_path])
        img_pair = torch.stack((torch.tensor(baseline_img, dtype=torch.float), torch.tensor(img, dtype=torch.float)), dim=0)
        final_label = torch.tensor(label, dtype=torch.float)

        return img_path_pair, img_pair, final_label
import os
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torch


class TestDatabase(Dataset):
    def __init__(self, root_dir, bp_folder, ap_folder, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.frame_paths = []    # List of all frame paths
        self.video_indices = []  # List to track which video each frame belongs to
        self.video_folders = []  # List to store names of video folders (for reference)
        self.labels = []  # List to store ap/bp labels

        # Scan root directory for subdirectories (each represents an AP video)
        subdirs_ap = sorted([os.path.join(os.path.join(root_dir, ap_folder), d) for d in os.listdir(os.path.join(root_dir, ap_folder)) if os.path.isdir(os.path.join(os.path.join(root_dir, ap_folder), d))])
        
        for video_idx, folder in enumerate(subdirs_ap):
            self.video_folders.append(folder)
            frame_files = sorted(os.listdir(folder))  # Sort to maintain frame order within each video
            frame_paths = [os.path.join(folder, f) for f in frame_files if f.endswith('.jpg') or f.endswith('.png')]
            self.frame_paths.extend(frame_paths)
            self.video_indices.extend([video_idx] * len(frame_paths))  # Track which video each frame belongs to
            self.labels.extend([0] * len(frame_paths))

        
        # Scan root directory for subdirectories (each represents a BP video)
        subdirs_bp = sorted([os.path.join(os.path.join(root_dir, bp_folder), d) for d in os.listdir(os.path.join(root_dir, bp_folder)) if os.path.isdir(os.path.join(os.path.join(root_dir, bp_folder), d))])
        
        for video_idx, folder in enumerate(subdirs_bp):
            self.video_folders.append(folder)
            frame_files = sorted(os.listdir(folder))  # Sort to maintain frame order within each video
            frame_paths = [os.path.join(folder, f) for f in frame_files if f.endswith('.jpg') or f.endswith('.png')]
            self.frame_paths.extend(frame_paths)
            self.video_indices.extend([video_idx + len(subdirs_ap)] * len(frame_paths))  # Track which video each frame belongs to
            self.labels.extend([1] * len(frame_paths))

    def __len__(self):
        return len(self.frame_paths)
    

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        frame = Image.open(frame_path).convert("RGB")  # Load image and convert to RGB
        if self.transform:
            frame = self.transform(frame)
        video_idx = self.video_indices[idx]
        return frame, video_idx, self.labels[idx]

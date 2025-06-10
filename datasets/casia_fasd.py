import os
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torch
import cv2
from torch.utils.data import WeightedRandomSampler


class CASIA_FASD(Dataset):
    def __init__(self, root, split="Train", bp_folder="real", ap_folder="attack", num_frames=25, transform=None):
        self.root = root
        self.transform = transform

        self.images, self.video_idx, self.labels = [], [], []
        
        videos_attacks = os.listdir(os.path.join(root, split, ap_folder))
        for i, v in enumerate(videos_attacks):
            frames_attacks = list(Path(os.path.join(root, split, ap_folder, v)).glob("*.[bjp][mpn][pg]"))
            frames_attacks.sort()
            num_frames_tmp = num_frames if len(frames_attacks) > num_frames else len(frames_attacks) 
            frames_attacks = frames_attacks[0:len(frames_attacks) - 1: len(frames_attacks)//num_frames_tmp]
            self.images = [*self.images, *frames_attacks]
            self.labels = [*self.labels, *[0]*len(frames_attacks)]
            self.video_idx.extend([i]*len(frames_attacks))

        videos_real = os.listdir(os.path.join(root, split, bp_folder))
        for i, v in enumerate(videos_real):
            frames_real = list(Path(os.path.join(root, split, bp_folder, v)).glob("*.[bjp][mpn][pg]"))
            frames_real.sort()
            num_frames_tmp = num_frames if len(frames_real) > num_frames else len(frames_real) 
            frames_real = frames_real[0:len(frames_real) - 1: len(frames_real)//num_frames_tmp]
            self.images = [*self.images, *frames_real]
            self.labels = [*self.labels, *[1]*len(frames_real)]
            self.video_idx.extend([i + len(videos_attacks)]*len(frames_real))


    def __len__(self):
        return len(self.images)


    def ApplyWeightedRandomSampler(self):
        class_counts = [len(self.labels) - np.sum(self.labels), np.count_nonzero(self.labels)]

        sample_weights = [1/class_counts[i] for i in self.labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(self.labels), replacement=True)
        return sampler


    def __getitem__(self, idx):
        image = cv2.imread(str(self.images[idx]))
        if image is None:
            raise Exception('Error: Image is None.')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image = image)['image']

        video_idx = self.video_idx[idx]
        return image, video_idx, torch.tensor(self.labels[idx], dtype=(torch.float32))

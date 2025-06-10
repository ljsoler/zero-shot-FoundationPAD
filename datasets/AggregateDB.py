import os
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torch
import cv2
from torch.utils.data import WeightedRandomSampler
from itertools import groupby


class AggregateDB(Dataset):
    def __init__(self, root, databases=["CASIA_FASD", "MSU-FASD", "RA"], num_frames=25, transform=None):
        self.root = root
        self.transform = transform

        self.images, self.video_idx, self.labels = [], [], []

        count_videos = 0
        for db in databases:
            videos = list(Path(os.path.join(root, db, 'frames')).rglob("*.[bjp][mpn][pg]"))
            videos.sort(key=lambda x: x.parent.name)  # groupby requires sorted data
            grouped = groupby(videos, key=lambda x: x.parent.name)
            for _, frames in grouped:
                frames = list(frames)
                frames.sort()
                num_frames_tmp = num_frames if len(frames) > num_frames else len(frames) 
                frames = frames[0:len(frames) - 1: len(frames)//num_frames_tmp]
                self.images.extend(frames)
                if 'OULU' in db:
                    self.labels.extend([1 if frames[0].parent.name.endswith("1") else 0]*len(frames))
                else:
                    self.labels.extend([1 if 'real' in str(frames[0]) else 0]*len(frames))
                self.video_idx.extend([count_videos]*len(frames))
                count_videos+=1


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

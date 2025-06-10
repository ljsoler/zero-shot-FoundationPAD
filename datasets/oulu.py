import os
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import cv2
import torch
import random
from torch.utils.data import WeightedRandomSampler

class OULU(Dataset):

    def __init__(self, root, split="Train", protocol = "1", num_frames=25, transform=None):

        self.root = root
        self.transform = transform

        file_protocol = os.path.join(root, 'Protocols', 'Protocol_{}'.format(protocol), '{}.txt'.format(split)) if not '3' in protocol and not '4' in protocol  else os.path.join(root, 'Protocols', 'Protocol_{}'.format(protocol.split('_')[0]), '{}_{}.txt'.format(split, protocol.split('_')[1]))

        self.images, self.video_idx, self.labels = [], [], []
        count_videos = 0
        with open(file_protocol) as f:
            for l in f.readlines():
                line = l.split(',')
                target = 1 if int(line[0]) > 0 else 0
                sample = line[1].strip('\n')
                #reading frames for the sample video
                frames = list(Path(os.path.join(root, split, sample)).glob("*.[bjp][mpn][pg]"))
                frames.sort()
                num_frames_tmp = num_frames if len(frames) > num_frames else len(frames) 
                frames = frames[0:len(frames) - 1: len(frames)//num_frames_tmp]
                self.labels = [*self.labels, *[target]*len(frames)]
                self.images = [*self.images, *frames]
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



class OULUTest(Dataset):

    def __init__(self, root, split="Test", protocol = "1", transform=None):

        self.root = root

        self.transform = transform

        file_protocol = os.path.join(root, 'Protocols', 'Protocol_{}'.format(protocol), '{}.txt'.format(split)) if not '3' in protocol and not '4' in protocol  else os.path.join(root, 'Protocols', 'Protocol_{}'.format(protocol.split('_')[0]), '{}_{}.txt'.format(split, protocol.split('_')[1]))

        self.images, self.labels, self.video_indices = [], [], []
        with open(file_protocol) as f:
            for video_idx, l in enumerate(f.readlines()):
                line = l.split(',')
                target = 1 if int(line[0]) > 0 else 0
                sample = line[1].strip('\n')
                #reading frames for the sample video
                frames = list(Path(os.path.join(root, split, sample)).glob("*.[bjp][mpn][pg]"))
                frames.sort()
                frames = [random.choice(frames)] #frames[0:len(frames) - 1:len(frames)//10]
                self.labels.extend([target]*len(frames))
                self.images.extend(frames)
                self.video_indices.extend([video_idx] * len(frames))


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        image = Image.open(str(self.images[idx]))

        if self.transform is not None:
            image = self.transform(image)

        video_idx = self.video_indices[idx]
        return image, video_idx, self.labels[idx]
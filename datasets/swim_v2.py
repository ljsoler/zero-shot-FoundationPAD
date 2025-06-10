import os
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torch
import cv2
from torch.utils.data import WeightedRandomSampler

Protocols = {
    "1": {
        "Train": ('trainlist_all.txt', 'trainlist_live.txt'),
        "Test": ('testlist_all.txt', 'testlist_live.txt')
    },
    "2": {
        "Train": 'trainlist_live.txt',
        "Test": 'testlist_live.txt'
    }
}

class SwimV2(Dataset):
    def __init__(self, root:str, split:str="Train", protocol:str = "1", test_attack:str = 'Makeup_Cosmetic', num_frames=25, transform=None):

        self.root = root
        self.transform = transform

        self.images, self.video_idx, self.labels = [], [], []
        protocol_split = Protocols[protocol][split]

        if isinstance(protocol_split, tuple): # this is for protocol 1
            file_protocol_attack = os.path.join(root, 'Protocols', protocol_split[0])
            file_protocol_bp = os.path.join(root, 'Protocols', protocol_split[1])

            count_videos = 0
            #loading bona fide
            with open(file_protocol_bp) as f:
                for line in f.readlines():
                    #reading frames for the sample video
                    frames = list(Path(os.path.join(root, 'live', line.strip('\n'))).glob("*.[bjp][mpn][pg]"))
                    if len(frames) > 0:
                        frames.sort()
                        num_frames_tmp = num_frames if len(frames) > num_frames else len(frames) 
                        frames = frames[0:len(frames) - 1: len(frames)//num_frames_tmp]
                        self.labels = [*self.labels, *[1]*len(frames)]
                        self.images = [*self.images, *frames]
                        self.video_idx.extend([count_videos]*len(frames))
                        count_videos+=1
            
            #loading attack presentations
            with open(file_protocol_attack) as f:
                for line in f.readlines():
                    #reading frames for the sample video
                    frames = list(Path(os.path.join(root, 'spoof_all', line.strip('\n'))).glob("*.[bjp][mpn][pg]"))
                    if len(frames) > 0:
                        frames.sort()
                        num_frames_tmp = num_frames if len(frames) > num_frames else len(frames) 
                        frames = frames[0:len(frames) - 1: len(frames)//num_frames_tmp]
                        self.labels = [*self.labels, *[0]*len(frames)]
                        self.images = [*self.images, *frames]
                        self.video_idx.extend([count_videos]*len(frames))
                        count_videos+=1

        else: # this is for protocol 2
            file_protocol_bp = os.path.join(root, 'Protocols', protocol_split)

            count_videos = 0
            #loading bona fide
            with open(file_protocol_bp) as f:
                for line in f.readlines():
                    #reading frames for the sample video
                    frames = list(Path(os.path.join(root, 'live', line.strip('\n'))).glob("*.[bjp][mpn][pg]"))
                    if len(frames) > 0:
                        frames.sort()
                        num_frames_tmp = num_frames if len(frames) > num_frames else len(frames) 
                        frames = frames[0:len(frames) - 1: len(frames)//num_frames_tmp]
                        self.labels = [*self.labels, *[1]*len(frames)]
                        self.images = [*self.images, *frames]
                        self.video_idx.extend([count_videos]*len(frames))
                        count_videos+=1

            #loading attack presentations
            attacks = [test_attack] if 'Test' in split else [f for f in os.listdir(os.path.join(root, 'spoof')) if not test_attack in f]

            for at in attacks:
                videos = os.listdir(os.path.join(root, 'spoof', at))
                for v in videos:
                    frames = list(Path(os.path.join(root, 'spoof', at, v)).glob("*.[bjp][mpn][pg]"))
                    if len(frames) > 0:
                        frames.sort()
                        num_frames_tmp = num_frames if len(frames) > num_frames else len(frames) 
                        frames = frames[0:len(frames) - 1: len(frames)//num_frames_tmp]
                        self.labels = [*self.labels, *[0]*len(frames)]
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


class SwimV2Test(Dataset):
    def __init__(self, root:str, split:str="Test", protocol:str = "2", test_attack:str = 'Makeup_Cosmetic', transform=None):

        self.root = root
        self.transform = transform

        self.images, self.labels, self.video_indices = [], [], []
        protocol_split = Protocols[protocol][split]

        if isinstance(protocol_split, tuple): # this is for protocol 1
            file_protocol_attack = os.path.join(root, 'Protocols', protocol_split[0])
            file_protocol_bp = os.path.join(root, 'Protocols', protocol_split[1])

            #loading bona fide
            with open(file_protocol_bp) as f:
                for line in f.readlines():
                    #reading frames for the sample video
                    frames = list(Path(os.path.join(root, 'live', line.strip('\n'))).glob("*.[bjp][mpn][pg]"))
                    self.labels = [*self.labels, *[1]*len(frames)]
                    self.images = [*self.images, *frames]
            
            #loading attack presentations
            with open(file_protocol_attack) as f:
                for line in f.readlines():
                    #reading frames for the sample video
                    frames = list(Path(os.path.join(root, 'spoof_all', line.strip('\n'))).glob("*.[bjp][mpn][pg]"))
                    self.labels.extend([0]*len(frames))
                    self.images.extend(frames)
                    

        else: # this is for protocol 2
            file_protocol_bp = os.path.join(root, 'Protocols', protocol_split)

            #loading bona fide
            with open(file_protocol_bp) as f:
                for video_idx, line in enumerate(f.readlines()):
                    #reading frames for the sample video
                    frames = list(Path(os.path.join(root, 'live', line.strip('\n'))).glob("*.[bjp][mpn][pg]"))
                    self.labels.extend([1]*len(frames))
                    self.images.extend(frames)
                    self.video_indices.extend([video_idx] * len(frames))

            #loading attack presentations
            attacks = [test_attack] if 'Test' in split else [f for f in os.listdir(os.path.join(root, 'spoof')) if not test_attack in f]
            max_indices = np.max(self.video_indices)
            for at in attacks:
                folder=os.path.join(root, 'spoof', at)
                subdirs_ap = sorted(os.listdir(folder))  # Sort to maintain frame order within each video
                for video_idx, s in enumerate(subdirs_ap):
                    temp_path = os.path.join(folder, s)
                    frame_files = sorted(os.listdir(temp_path))
                    frame_paths = [os.path.join(temp_path, f) for f in frame_files if f.endswith('.jpg') or f.endswith('.png')]
                    self.labels.extend([0]*len(frame_files))
                    self.images.extend(frame_paths)
                    self.video_indices.extend([video_idx + max_indices] * len(frame_paths))  # Track which video each frame belongs to


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(str(self.images[idx])).convert("RGB")  # Load image and convert to RGB

        if self.transform is not None:
            image = self.transform(image)

        video_idx = self.video_indices[idx]
        return image, video_idx, self.labels[idx]
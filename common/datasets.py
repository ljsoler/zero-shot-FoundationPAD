import numpy as np
import cv2
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image 


def load_dataset(input_path, ext, transformation):

    files = list(Path(input_path).rglob("**/*{}".format(ext)))

    labels = np.array([1 if 'live' in str(e) else 0 for e in files])

    images = list([Image.open(str(s)) for s in files])

    images = torch.cat([transformation(img).unsqueeze(0) for img in images])

    return images, labels



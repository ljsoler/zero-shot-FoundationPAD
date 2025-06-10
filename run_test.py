import yaml, os
import argparse, csv
import numpy as np
from models.DNNBackbone import DNNBackbone 
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from datasets import *
from collections import defaultdict
from tqdm import tqdm
from callbacks.DetectionPerformanceCallback import DetectionPerformanceCallback
from datasets import *
import albumentations
from albumentations.pytorch import ToTensorV2

torch.set_float32_matmul_precision('medium')

parser = argparse.ArgumentParser(description='Transfer learning for PAD')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='./configs/config_casia_clip.yaml')

parser.add_argument('--ap_folder',  '-a',
                    help =  'folder name in the root folder containing the AP videos',
                    default='Makeup_Cosmetic')

parser.add_argument('--bp_folder',  '-b',
                    help =  'folder name in the root folder containing the BP videos',
                    default='real')

parser.add_argument('--model_weights',  '-m',
                    help =  'path to the model weights',
                    default='logs/OULU/dinov2_vits14_OULU/Protocol_1/epoch_29-D-EER_5.79.ckpt')

parser.add_argument('--output_folder',  '-o',
                    help =  'path to the output folder',
                    default='logs/OULU/dinov2_vits14_OULU')

args = parser.parse_args()

print('Loading configuration file -> {}'.format(args.filename))

with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

transform_test = albumentations.Compose([
            albumentations.Resize(height=config['dataset']['img_size'], width=config['dataset']['img_size']),
            albumentations.Normalize(**config['dataset']['normalize'], always_apply=True),
            ToTensorV2(),
        ])

#loading the database
test_dataset = DATASET[config['dataset']['name']](**config['dataset']['test_params'], bp_folder=args.bp_folder, ap_folder=args.ap_folder, transform=transform_test)

test_dataloader = DataLoader(test_dataset, batch_size=config['dataset']['batch_size'],
                            num_workers=config['dataset']['num_workers'], shuffle=False)

#loading model
model = DNNBackbone.load_from_checkpoint(args.model_weights, **config['model_params'])
model.eval()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

os.makedirs(args.output_folder, exist_ok=True)

performance_callback = DetectionPerformanceCallback(test_dataloader, output_folder=args.output_folder)

callbacks = [performance_callback]

trainer = Trainer(callbacks=callbacks,
                 enable_progress_bar=True,
                 **config['trainer_params'])

result = trainer.test(model=model, dataloaders=test_dataloader, ckpt_path=args.model_weights)
    
with open(os.path.join(args.output_folder, 'metric_info.csv'), mode='w') as csv_file:
    fieldnames = ['D-EER_Frames', 'BPCER10_Frames', 'BPCER20_Frames', 'BPCER100_Frames', 'D-EER_Videos', 'BPCER10_Videos', 'BPCER20_Videos', 'BPCER100_Videos', 'test_acc', 'test_loss']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(result[0])

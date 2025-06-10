import yaml, os
import argparse, csv
import numpy as np
from models.DNNBackbone import DNNBackbone 
import torch
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from callbacks.DetectionPerformanceCallback import DetectionPerformanceCallback
from torch.utils.data import DataLoader
from datasets import *

import albumentations
from albumentations.pytorch import ToTensorV2

torch.set_float32_matmul_precision('medium')

parser = argparse.ArgumentParser(description='Transfer learning for PAD')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='./configs/config_cross_db.yaml')

args = parser.parse_args()

print('Loading configuration file -> {}'.format(args.filename))

with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

transform_train = albumentations.Compose([
            albumentations.Resize(height=256, width=256),
            albumentations.RandomCrop(height=config['dataset']['img_size'], width=config['dataset']['img_size']),
            albumentations.HorizontalFlip(),
            albumentations.RandomGamma(gamma_limit=(80, 180)), # 0.5, 1.5
            albumentations.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
            albumentations.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            albumentations.Normalize(**config['dataset']['normalize'], always_apply=True),
            ToTensorV2()
        ])
        
transform_val = albumentations.Compose([
            albumentations.Resize(height=config['dataset']['img_size'], width=config['dataset']['img_size']),
            albumentations.Normalize(**config['dataset']['normalize'], always_apply=True),
            ToTensorV2(),
        ])
        
#loading the database
train_dataset = DATASET[config['dataset']['name']](**config['dataset']['train_params'], transform=transform_train)
val_dataset = DATASET[config['dataset']['name']](**config['dataset']['val_params'], transform=transform_val)

train_dataloader = DataLoader(train_dataset, batch_size=config['dataset']['batch_size'],
                              num_workers=config['dataset']['num_workers'], drop_last=True, sampler=train_dataset.ApplyWeightedRandomSampler())
val_dataloader = DataLoader(val_dataset, batch_size=config['dataset']['batch_size'],
                              num_workers=config['dataset']['num_workers'])

# Initialise callbacks
monitor = 'D-EER'
filename = '{epoch}-{D-EER:.2f}'

checkpoint_callback = ModelCheckpoint(
    monitor=monitor,
    dirpath=os.path.join(config['logging_params']['run_dir'], '{}_{}'.format(config['model_params']['model_name'], config['dataset']['name']), config['logging_params']['version']),
    filename=filename
)
performance_callback = DetectionPerformanceCallback(val_dataloader, output_folder=os.path.join(config['logging_params']['run_dir'], '{}_{}'.format(config['model_params']['model_name'], config['dataset']['name']), config['logging_params']['version']))

callbacks = [checkpoint_callback, performance_callback]

logger = TensorBoardLogger(config['logging_params']['run_dir'], name='{}_{}'.format(config['model_params']['model_name'], config['dataset']['name']),
                           version=config['logging_params']['version'] if 'version' in config['logging_params'] else None)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

#loading model
model = DNNBackbone(**config['model_params'])

checkpoint_resume = config['dataset']['checkpoint'] if config['dataset']['checkpoint'] != 'None' else None 

trainer = Trainer(logger=logger,
                 callbacks=callbacks,
                 enable_progress_bar=True,
                 **config['trainer_params'])

print(f"======= Training {config['model_params']['model_name']} =======")
trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=checkpoint_resume)

if 'test_params' in config['dataset']:
    test_dataset = DATASET[config['dataset']['name']](**config['dataset']['test_params'], transform=transform_val)
    test_dataloader = DataLoader(test_dataset, batch_size=config['dataset']['batch_size'],
                              num_workers=config['dataset']['num_workers'])

    result = trainer.test(model=model, dataloaders=test_dataloader, ckpt_path="best")
    
    with open(os.path.join(os.path.join(config['logging_params']['run_dir'], '{}_{}'.format(config['model_params']['model_name'], config['dataset']['name']), config['logging_params']['version']), 'metric_info.csv'), mode='w') as csv_file:
        fieldnames = ['D-EER_Frames', 'BPCER10_Frames', 'BPCER20_Frames', 'BPCER100_Frames', 'D-EER_Videos', 'BPCER10_Videos', 'BPCER20_Videos', 'BPCER100_Videos', 'test_acc', 'test_loss']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(result[0])

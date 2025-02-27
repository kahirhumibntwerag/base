import os
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import wandb
import yaml
from pathlib import Path
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import argparse
from omegaconf import OmegaConf
from inference import Model
from src.RRDB import LightningGenerator
from src.datamodule import DataModule

def rescalee(images):
    images_clipped = torch.clamp(images, min=1)
    images_log = torch.log(images_clipped)
    max_value = torch.log(torch.tensor(20000))
    max_value = torch.clamp(max_value, min=1e-9)
    images_normalized = images_log / max_value
    return images_normalized

def inverse_rescalee(images_normalized):
    max_value = torch.log(torch.tensor(20000.0))
    max_value = torch.clamp(max_value, min=1e-9)
    images_log = images_normalized * max_value
    images_clipped = torch.exp(images_log)

    return images_clipped

def parse_and_merge_config(config_path='config.yml'):
    # Create a parser to accept arguments from the command line
    parser = argparse.ArgumentParser(description="Train the model with optional config overrides.")
    
    # Parse known arguments and handle unknown arguments separately
    parser.add_argument('--config', type=str, default=config_path, help='Path to config file')
    parser.add_argument('--model_name', type=str, default='rrdb', help='model name')
    parser.add_argument('--model_path', type=str, default='checkpoints/model.pt', help='model path')

    parser.add_argument('--opt', nargs='+', default=None, help='Override config options, e.g., data.batch_size=64')
    args = parser.parse_args()
    
    # Load the configuration from the YAML file
    with open(args.config, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Convert the loaded YAML config into an OmegaConf object
    cfg = OmegaConf.create(yaml_config)
    
    # If there are command-line options, merge them with the config
    if args.opt:
        cli_conf = OmegaConf.from_dotlist(args.opt)
        cfg = OmegaConf.merge(cfg, cli_conf)
    
    # Return the final configuration
    return cfg

def train():
    # Parse and merge config
    config = parse_and_merge_config()
    
    # Initialize wandb logger
    wandb_logger = WandbLogger(
        **OmegaConf.to_container(config.logger),
        config=OmegaConf.to_container(config)
    )

    # Define transforms with power transform instead of rescalee
    transform = transforms.Compose([
        rescalee
    ])
    
    # Initialize DataModule and Model with transforms
    datamodule = DataModule(**OmegaConf.to_container(config.data), transform=transform)
    model = Model().instantiate_model(config.model_name,config.model_path)
    wandb_logger.watch(model, log='all')
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(**OmegaConf.to_container(config.callbacks.checkpoint)),
    ]

    # Initialize Trainer
    trainer = L.Trainer(
        **OmegaConf.to_container(config.trainer),
        logger=wandb_logger,
        callbacks=callbacks
    )

    # Train the model
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    train()

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
from src.datamodule import DataModule
from src.callbacks import ImageLoggingCallback

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['rrdb', 'esrgan'],
                      help='Model architecture to use')
    parser.add_argument('--config', type=str, default=None,
                      help='Optional path to override default configs')
    parser.add_argument('--opt', nargs='+', default=None,
                      help='Override config options (e.g., trainer.max_epochs=100)')
    return parser.parse_args()

def load_and_merge_configs(args):
    """
    Load and merge configurations from all sources:
    1. Base config (config.yml)
    2. Model-specific config (models/{model}.yml)
    3. Custom config file (optional)
    4. Command line overrides (--opt)
    """
    # Load base config
    base_config = OmegaConf.load('configs/config.yml')
    
    # Load model-specific config
    model_config_path = Path(f'configs/models/{args.model}.yml')
    if not model_config_path.exists():
        raise ValueError(f"No config found for model: {args.model}")
    model_config = OmegaConf.load(model_config_path)
    
    # Merge base and model configs
    config = OmegaConf.merge(base_config, model_config)
    
    # Merge custom config if provided
    if args.config:
        custom_config = OmegaConf.load(args.config)
        config = OmegaConf.merge(config, custom_config)
    
    # Apply command-line overrides
    if args.opt:
        cli_config = OmegaConf.from_dotlist(args.opt)
        config = OmegaConf.merge(config, cli_config)
    
    return config

def train():
    # Parse command line arguments
    args = parse_args()
    
    # Load and merge all configs
    config = load_and_merge_configs(args)
    
    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project=config.logger.project,
        name=f"{args.model}-{wandb.util.generate_id()}",
        config=OmegaConf.to_container(config)
    )

    # Initialize callbacks
    callbacks = [
        ModelCheckpoint(**config.callbacks.checkpoint),
        ImageLoggingCallback(**config.callbacks.image_logger)
    ]

    # Initialize trainer
    trainer = L.Trainer(
        **config.trainer,
        logger=wandb_logger,
        callbacks=callbacks
    )
    model = Model().instantiate_model(config.model)

    datamodule = DataModule(**config.data)

    # Train model
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    train()

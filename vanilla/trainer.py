from tqdm.notebook import trange, tqdm
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class Trainer:
    def __init__(self, max_epochs, device, callbacks=None):
        self.max_epochs = max_epochs
        self.current_epoch = 0
        self.global_step = 0
        self.callbacks = callbacks or []
        self.device = device

    def _run_callback_hook(self, hook_name, *args, **kwargs):
        """Generic method to run callback hooks safely"""
        for callback in self.callbacks:
            if hasattr(callback, hook_name) and callable(getattr(callback, hook_name)):
                getattr(callback, hook_name)(self, *args, **kwargs)

    def _train_epoch(self, model, train_loader):
        """Handle single training epoch"""
        self._run_callback_hook('on_train_epoch_start', model)
        model.train()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, leave=False)):
            self._train_step(model, batch, batch_idx)
        
        self._run_callback_hook('on_train_epoch_end', model)

    def _train_step(self, model, batch, batch_idx):
        """Handle single training step"""
        self._run_callback_hook('on_train_batch_start', model, batch, batch_idx)
        
        batch = tuple([tensor.to(self.device) for tensor in batch])
        model.training_step(batch)
        self.global_step += 1
        
        self._run_callback_hook('on_train_batch_end', model, batch, batch_idx)

    def _validation_epoch(self, model, val_loader):
        """Handle single validation epoch"""
        self._run_callback_hook('on_validation_epoch_start', model)
        model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, leave=False)):
                self._validation_step(model, batch, batch_idx)
        
        self._run_callback_hook('on_validation_epoch_end', model)

    def _validation_step(self, model, batch, batch_idx):
        """Handle single validation step"""
        self._run_callback_hook('on_validation_batch_start', model, batch, batch_idx)
        
        batch = tuple([tensor.to(self.device) for tensor in batch])
        model.validation_step(batch)
        
        self._run_callback_hook('on_validation_batch_end', model, batch, batch_idx)

    def _setup_training(self, model, datamodule):
        """Setup training environment"""
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        model.to(self.device)
        return train_loader, val_loader

    def fit(self, model, datamodule):
        """Main training loop with hooks"""
        try:
            self._run_callback_hook('on_fit_start', model)
            train_loader = datamodule.train_dataloader()
            val_loader = datamodule.val_dataloader()
            model.to(self.device)

            for epoch in trange(self.current_epoch, self.max_epochs):
                self.current_epoch = epoch
                
                # Training phase
                self._train_epoch(model, train_loader)
                
                # Validation phase
                self._validation_epoch(model, val_loader)

            self._run_callback_hook('on_fit_end', model)

        except Exception as e:
            self._run_callback_hook('on_exception', model, e)
            raise e

    # Hook methods
    def on_fit_start(self, model, datamodule):
        """Called when fit begins"""
        self._run_callback_hook('on_fit_start', model)

    def on_fit_end(self, model, datamodule):
        """Called when fit ends"""
        self._run_callback_hook('on_fit_end', model)

    def on_train_epoch_start(self, model):
        """Called when training epoch begins"""
        self._run_callback_hook('on_train_epoch_start', model)

    def on_train_batch_start(self, model, batch, batch_idx):
        """Called before training batch"""
        self._run_callback_hook('on_train_batch_start', model, batch, batch_idx)

    def on_train_batch_end(self, model, batch, batch_idx):
        """Called after training batch"""
        self._run_callback_hook('on_train_batch_end', model, batch, batch_idx)

    def on_train_epoch_end(self, model, train_loader):
        """Called when training epoch ends"""
        self._run_callback_hook('on_train_epoch_end', model)

    def on_validation_epoch_start(self, model):
        """Called when validation epoch begins"""
        self._run_callback_hook('on_validation_epoch_start', model)

    def on_validation_batch_start(self, model, batch, batch_idx):
        """Called before validation batch"""
        self._run_callback_hook('on_validation_batch_start', model, batch, batch_idx)

    def on_validation_batch_end(self, model, batch, batch_idx):
        """Called after validation batch"""
        self._run_callback_hook('on_validation_batch_end', model, batch, batch_idx)

    def on_validation_epoch_end(self, model, val_loader):
        """Called when validation epoch ends"""
        self._run_callback_hook('on_validation_epoch_end', model)

    def on_exception(self, model, exception):
        """Called when an exception occurs"""
        self._run_callback_hook('on_exception', model, exception)




class Module(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.logged_metrics = {}
    
    @abstractmethod
    def training_step(self, batch):
        """
        Perform a training step.
        Args:
            batch: The input batch of data
        """
        raise NotImplementedError("training_step must be implemented")

    @abstractmethod
    def validation_step(self, batch):
        """
        Perform a validation step.
        Args:
            batch: The input batch of data
        """
        raise NotImplementedError("validation_step must be implemented")

    @abstractmethod
    def configure_optimizers(self):
        """
        Configure and return optimizers and learning rate schedulers.
        Returns:
            tuple: (optimizer, scheduler) or optimizer
        """
        raise NotImplementedError("configure_optimizers must be implemented")

    def log(self, *args, **kwargs):
        if args[0] not in self.logged_metrics:
            self.logged_metrics[args[0]] = []
        self.logged_metrics[args[0]] += [args[1].item()]
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import lightning as L
import wandb
import matplotlib.pyplot as plt

class ResidualBlock(nn.Module):
    """
    Define a Residual Block without Batch Normalization
    """
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class RRDB(nn.Module):
    """
    Define the Residual in Residual Dense Block (RRDB)
    """
    def __init__(self, in_features, num_dense_layers=3):
        super(RRDB, self).__init__()
        self.residual_blocks = nn.Sequential(*[ResidualBlock(in_features) for _ in range(num_dense_layers)])

    def forward(self, x):
        return x + self.residual_blocks(x)


class Generator(nn.Module):
    """
    Define the Generator network for solar images with 1 channel
    """
    def __init__(self, in_channels=1, initial_channel=64, num_rrdb_blocks=4, upscale_factor=4, lr=1e-4, **kwargs):
        super(Generator, self).__init__()

        self.lr = lr

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, initial_channel, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        # RRDB blocks
        self.rrdbs = nn.Sequential(*[RRDB(initial_channel) for _ in range(num_rrdb_blocks)])

        # Post-residual blocks
        self.post_rrdb = nn.Sequential(
            nn.Conv2d(initial_channel, initial_channel, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )

        # Upsampling layers
        self.upsampling = nn.Sequential(
            *[nn.Conv2d(initial_channel, 4*initial_channel, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()]*int(np.log2(upscale_factor)))
        # Output layer
        self.output = nn.Conv2d(initial_channel, in_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        initial = self.initial(x)
        rrdbs = self.rrdbs(initial)
        post_rrdb = self.post_rrdb(rrdbs + initial)
        upsampled = self.upsampling(post_rrdb)
        return self.output(upsampled)


class LightningGenerator(Module):
    def __init__(self, config):
        super(LightningGenerator, self).__init__()
        self.generator = Generator(**config['generator'])
        self.opt = self.configure_optimizers()

    def forward(self, x):
        return self.generator(x)
    
    def training_step(self, batch):
        lr, hr = batch
        sr = self(lr)
        loss = F.mse_loss(sr, hr)
        self.log('train_loss', loss)
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
    
    def validation_step(self, batch):
        lr, hr = batch
        sr = self(lr)
        loss = F.mse_loss(sr, hr)
        self.log('val_loss', loss)      

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.generator.lr)
        return optimizer
    
    def predict(self, lr):
        return self(lr)
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, config=None, strict=True, **kwargs):
        """
        Load model from checkpoint with optional config override
        Args:
            checkpoint_path (str): Path to the checkpoint file
            config (dict, optional): New configuration to override the saved one
            strict (bool): Whether to strictly enforce that the keys in checkpoint match
        """
        if config is not None:
            # Load the checkpoint with custom config
            checkpoint = torch.load(checkpoint_path)
            if 'hyper_parameters' in checkpoint:
                checkpoint['hyper_parameters'] = config
            return super().load_from_checkpoint(checkpoint_path, strict=strict, **kwargs)
        else:
            # Load the checkpoint normally
            return super().load_from_checkpoint(checkpoint_path, strict=strict, **kwargs)


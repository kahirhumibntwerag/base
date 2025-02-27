import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import lightning as L
import wandb
import matplotlib.pyplot as plt
from discriminator import Discriminator
from generator import Generator
from loss import Loss
import lightning as L

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



class GAN(L.LightningModule):
    def __init__(self, configs):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.generator = Generator(**configs['generator'])
        self.discriminator = Discriminator(**configs['discriminator'])
        self.loss = Loss(**configs['loss'])
        
        # Add label smoothing parameters
        self.real_label_val = configs['label_smoothing']['real_val']  # Instead of 1.0
        self.fake_label_val = configs['label_smoothing']['fake_val']  # Instead of 0.0

    def training_step(self, batch, batch_idx):
        opt_g, opt_disc = self.optimizers()
        
        lr, hr = batch
        sr = self.generator(lr)
        
        ###### discriminator #######
        # Real images with smoothed labels
        logits_real = self.discriminator(hr.contiguous().detach())
        real_labels = torch.ones_like(logits_real, device=logits_real.device) * self.real_label_val  # Smoothed to 0.9
        d_loss_real = self.loss.adversarial_loss(logits_real, real_labels)

        # Fake images with smoothed labels
        logits_fake = self.discriminator(sr.contiguous().detach())
        fake_labels = torch.zeros_like(logits_fake, device=logits_fake.device) + self.fake_label_val  # Smoothed to 0.1
        d_loss_fake = self.loss.adversarial_loss(logits_fake, fake_labels)
        
        d_loss = (d_loss_real + d_loss_fake) / 2
        self.log('d_loss', d_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        opt_disc.zero_grad()
        self.manual_backward(d_loss)
        opt_disc.step()
        
        ###### generator #######
        # For generator training, we still use 1.0 (no smoothing)
        logits_fake = self.discriminator(sr)
        real_labels = torch.ones_like(logits_fake, device=logits_fake.device)  # Keep at 1.0 for generator
        g_loss = self.loss.adversarial_loss(logits_fake, real_labels)
        adversarial_component = self.loss.adversarial_weight * g_loss
        self.log('train_g_loss', g_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        l1_loss = self.loss.l1_loss(hr, sr)
        l1_component = self.loss.l1_weight * l1_loss
        self.log('train_l1_loss', l1_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        perceptual_loss = torch.mean(self.loss.perceptual_loss(hr, sr))
        perceptual_component = self.loss.perceptual_weight * perceptual_loss
        self.log('train_perceptual_loss', perceptual_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        loss = perceptual_component + l1_component + adversarial_component 

        opt_g.zero_grad()
        self.manual_backward(loss)
        opt_g.step()

    def validation_step(self, x, batch_idx):
        lr, hr = x
        sr = self.generator(lr)
        
        logits_fake = self.discriminator(sr)
        real_labels = torch.ones_like(logits_fake, device=logits_fake.device)
        g_loss = self.loss.adversarial_loss(logits_fake, real_labels)
        self.log('val_g_loss', g_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        l1_loss = self.loss.l1_loss(hr, sr)
        self.log('val_l1_loss', l1_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        perceptual_loss = torch.mean(self.loss.perceptual_loss(hr, sr))  
        self.log('val_perceptual_loss', perceptual_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.discriminator.lr, betas=(0.5, 0.9)) 
        vae_opt = torch.optim.Adam(self.generator.parameters(), lr=self.generator.lr, betas=(0.5, 0.9)) 
        return [vae_opt, disc_opt]
    
    def predict(self, x):
        return self.generator(x)
    
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




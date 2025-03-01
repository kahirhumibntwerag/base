import torch
import torch.nn as nn
import math
import lightning as L
from torch.optim import AdamW
from src.ldm.vae.vae import VAEGAN
from src.ldm.ldm.diffusion import Diffusion
from src.ldm.ldm.unet import Unet
from src.ldm.ldm.sampler import Sampler
class LDM(L.LightningModule):
    def __init__(self, configs):
        super().__init__()
        self.save_hyperparameters()
        self.vae = VAEGAN.load_from_checkpoint(configs['vae_path']).vae.eval()
        self.unet = Unet(**configs['unet'])
        self.loss = nn.MSELoss()
        self.diffusion = Diffusion()
        self.sampler = Sampler()
        for param in self.vae.parameters():
            param.requires_grad = False
    
    def training_step(self, batch):
        lr, hr = batch
        z, _, _ = self.vae.encoder(hr)

        t = torch.randint(1000, size=(z.shape[0],), device=z.device)
        zt, noise = self.diffusion.add_noise(z, t)
        predicted_noise = self.unet(torch.cat([zt, lr], dim=1), t)
        loss = self.loss(noise, predicted_noise)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        lr, hr = batch
        z, _, _ = self.vae.encoder(hr)

        t = torch.randint(1000, size=(z.shape[0],), device=z.device)
        zt, noise = self.diffusion.add_noise(z, t)
        predicted_noise = self.unet(torch.cat([zt, lr], dim=1), t)
        loss = self.loss(noise, predicted_noise)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

        
    def configure_optimizers(self):
        return AdamW(self.unet.parameters(), lr=self.unet.lr)
    
    
    def predict(self, x, steps=50):
      sr = self.sampler.sample(x, steps, self.unet, self.vae)
      return sr
    
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


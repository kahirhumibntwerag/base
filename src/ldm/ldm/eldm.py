import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import lightning as L
from src.ldm.vae.vae import VAEGAN
import matplotlib.pyplot as plt
from src.gan.discriminator import DiscriminatorSRGAN
from src.ldm.vae.loss import Loss
from src.ldm.ldm.sampler import Sampler
import lightning as L
from src.ldm.ldm.ldm import LDM
def kl_loss(mu, logvar):
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
def get_norm_layer(channels, norm_type="bn"):
    if norm_type == "bn":
        return nn.BatchNorm2d(channels, eps=1e-4)
    elif norm_type == "gn":
        return nn.GroupNorm(8, channels, eps=1e-4)
    else:
        raise ValueError("norm_type must be bn or gn")


class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, norm_type="bn"):
        super(ResDown, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)

        self.conv1 = nn.Conv2d(channel_in, (channel_out // 2) + channel_out, kernel_size, 2, kernel_size // 2)
        self.norm2 = get_norm_layer(channel_out // 2, norm_type=norm_type)

        self.conv2 = nn.Conv2d(channel_out // 2, channel_out, kernel_size, 1, kernel_size // 2)

        self.act_fnc = nn.ELU()
        self.channel_out = channel_out

    def forward(self, x):
        x = self.act_fnc(self.norm1(x))

        # Combine skip and first conv into one layer for speed
        x_cat = self.conv1(x)
        skip = x_cat[:, :self.channel_out]
        x = x_cat[:, self.channel_out:]

        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)

        return x + skip


class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, scale_factor=2, norm_type="bn"):
        super(ResUp, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)

        self.conv1 = nn.Conv2d(channel_in, (channel_in // 2) + channel_out, kernel_size, 1, kernel_size // 2)
        self.norm2 = get_norm_layer(channel_in // 2, norm_type=norm_type)

        self.conv2 = nn.Conv2d(channel_in // 2, channel_out, kernel_size, 1, kernel_size // 2)

        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.act_fnc = nn.ELU()
        self.channel_out = channel_out

    def forward(self, x_in):
        x = self.up_nn(self.act_fnc(self.norm1(x_in)))

        # Combine skip and first conv into one layer for speed
        x_cat = self.conv1(x)
        skip = x_cat[:, :self.channel_out]
        x = x_cat[:, self.channel_out:]

        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)

        return x + skip


class ResBlock(nn.Module):
    """
    Residual block
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, norm_type="bn"):
        super(ResBlock, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)

        first_out = channel_in // 2 if channel_in == channel_out else (channel_in // 2) + channel_out
        self.conv1 = nn.Conv2d(channel_in, first_out, kernel_size, 1, kernel_size // 2)

        self.norm2 = get_norm_layer(channel_in // 2, norm_type=norm_type)

        self.conv2 = nn.Conv2d(channel_in // 2, channel_out, kernel_size, 1, kernel_size // 2)
        self.act_fnc = nn.ELU()
        self.skip = channel_in == channel_out
        self.bttl_nk = channel_in // 2

    def forward(self, x_in):
        x = self.act_fnc(self.norm1(x_in))

        x_cat = self.conv1(x)
        x = x_cat[:, :self.bttl_nk]

        # If channel_in == channel_out we do a simple identity skip
        if self.skip:
            skip = x_in
        else:
            skip = x_cat[:, self.bttl_nk:]

        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)

        return x + skip


class Encoder(nn.Module):
    """
    Encoder block
    """

    def __init__(self, channels, ch=64, blocks=[1, 2, 4, 8], latent_channels=256, num_res_blocks=1, norm_type="bn",
                 deep_model=False):
        super(Encoder, self).__init__()
        self.conv_in = nn.Conv2d(channels, blocks[0] * ch, 3, 1, 1)

        widths_in = blocks
        widths_out = blocks[1:] + [2 * blocks[-1]]

        self.layer_blocks = nn.ModuleList([])
        for w_in, w_out in zip(widths_in, widths_out):

            if deep_model:
                # Add an additional non down-sampling block before down-sampling
                self.layer_blocks.append(ResBlock(w_in * ch, w_in * ch, norm_type=norm_type))

            self.layer_blocks.append(ResDown(w_in * ch, w_out * ch, norm_type=norm_type))

        for _ in range(num_res_blocks):
            self.layer_blocks.append(ResBlock(widths_out[-1] * ch, widths_out[-1] * ch, norm_type=norm_type))

        self.conv_mu = nn.Conv2d(widths_out[-1] * ch, latent_channels, 1, 1)
        self.conv_log_var = nn.Conv2d(widths_out[-1] * ch, latent_channels, 1, 1)
        self.act_fnc = nn.ELU()

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, sample=False):
        x = self.conv_in(x)

        for block in self.layer_blocks:
            x = block(x)
        x = self.act_fnc(x)

        mu = self.conv_mu(x)
        log_var = self.conv_log_var(x)

        if self.training or sample:
            x = self.sample(mu, log_var)
        else:
            x = mu

        return x, mu, log_var


class Decoder(nn.Module):
    """
    Decoder block
    Built to be a mirror of the encoder block
    """

    def __init__(self, channels, ch=64, blocks=[1, 2, 4, 8], latent_channels=256, num_res_blocks=1, norm_type="bn",
                 deep_model=False):
        super(Decoder, self).__init__()
        widths_out = blocks[::-1]
        widths_in = (blocks[1:] + [2 * blocks[-1]])[::-1]

        self.conv_in = nn.Conv2d(latent_channels, widths_in[0] * ch, 1, 1)

        self.layer_blocks = nn.ModuleList([])
        for _ in range(num_res_blocks):
            self.layer_blocks.append(ResBlock(widths_in[0] * ch, widths_in[0] * ch, norm_type=norm_type))

        for w_in, w_out in zip(widths_in, widths_out):
            self.layer_blocks.append(ResUp(w_in * ch, w_out * ch, norm_type=norm_type))
            if deep_model:
                # Add an additional non up-sampling block after up-sampling
                self.layer_blocks.append(ResBlock(w_out * ch, w_out * ch, norm_type=norm_type))

        self.conv_out = nn.Conv2d(blocks[0] * ch, channels, 5, 1, 2)
        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.conv_in(x)

        for block in self.layer_blocks:
            x = block(x)
        x = self.act_fnc(x)

        return torch.tanh(self.conv_out(x))


class VAE(nn.Module):
    """
    VAE network, uses the above encoder and decoder blocks
    """
    def __init__(self, channel_in=1, ch=64, blocks=[1, 2], latent_channels=3, num_res_blocks=8, norm_type="gn",
                 deep_model=False, lr=0.0001):
        super(VAE, self).__init__()
        """Res VAE Network
        channel_in  = number of channels of the image
        z = the number of channels of the latent representation
        (for a 64x64 image this is the size of the latent vector)
        """
        self.lr = lr
        self.encoder = Encoder(channel_in, ch=ch, blocks=blocks, latent_channels=latent_channels,
                               num_res_blocks=num_res_blocks, norm_type=norm_type, deep_model=deep_model)
        self.decoder = Decoder(channel_in, ch=ch, blocks=blocks, latent_channels=latent_channels,
                               num_res_blocks=num_res_blocks, norm_type=norm_type, deep_model=deep_model)

    def forward(self, x):
        encoding, mu, log_var = self.encoder(x)
        recon_img = self.decoder(encoding)
        return recon_img, mu, log_var



class ELDM(L.LightningModule):
    def __init__(self, configs):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.ldm = LDM.load_from_checkpoint(configs['ldm_path'])
        self.vae = self.ldm.vae
        self.unet = self.ldm.unet
        self.sampler = Sampler()
        self.discriminator = DiscriminatorSRGAN(**configs['discriminator'])
        self.loss = Loss(self.discriminator,**configs['loss'])
        # Add label smoothing parameters
        self.real_label_val = configs['label_smoothing']['real_val']  # Instead of 1.0
        self.fake_label_val = configs['label_smoothing']['fake_val']  # Instead of 0.0
        # Add warmup steps
        self.disc_start_step = configs.get('disc_start_step', 0)  # Default to 50k steps if not specified
        self.global_step = 0
        for param in self.unet.parameters():
            param.requires_grad = False
        for param in self.vae.encoder.parameters():
            param.requires_grad = False
    
    def training_step(self, batch, batch_idx):
        opt_g, opt_disc = self.optimizers()
        self.global_step += 1
        
        lr, hr = batch
        decoded = self.predict(lr)
        
        # Only train discriminator after warmup period
        if self.global_step >= self.disc_start_step:
            ###### discriminator #######
            # Real images with smoothed labels
            logits_real = self.discriminator(hr.contiguous().detach())
            real_labels = torch.ones_like(logits_real, device=logits_real.device) * self.real_label_val
            d_loss_real = self.loss.adversarial_loss(logits_real, real_labels)

            # Fake images with smoothed labels
            logits_fake = self.discriminator(decoded.contiguous().detach())
            fake_labels = torch.zeros_like(logits_fake, device=logits_fake.device) + self.fake_label_val
            d_loss_fake = self.loss.adversarial_loss(logits_fake, fake_labels)
            
            d_loss = (d_loss_real + d_loss_fake) / 2
            self.log('d_loss', d_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

            opt_disc.zero_grad()
            self.manual_backward(d_loss)
            opt_disc.step()
        
        ###### generator #######
        # Only include adversarial loss after warmup period
        if self.global_step >= self.disc_start_step:
            logits_fake = self.discriminator(decoded)
            real_labels = torch.ones_like(logits_fake, device=logits_fake.device)
            g_loss = self.loss.adversarial_loss(logits_fake, real_labels)
            adversarial_component = self.loss.adversarial_weight * g_loss
            self.log('train_g_loss', g_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        else:
            adversarial_component = 0.0
        
        # Rest of the losses remain unchanged
        l1_loss = self.loss.l1_loss(hr, decoded)
        l1_component = self.loss.l1_weight * l1_loss
        self.log('train_l1_loss', l1_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        perceptual_loss = torch.mean(self.loss.perceptual_loss(hr, decoded))
        perceptual_component = self.loss.perceptual_weight * perceptual_loss
        self.log('train_perceptual_loss', perceptual_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)


        loss = perceptual_component + l1_component + adversarial_component

        opt_g.zero_grad()
        self.manual_backward(loss)
        opt_g.step()

    def validation_step(self, x, batch_idx):
        lr, hr = x
        decoded = self.predict(lr)
        
        logits_fake = self.discriminator(decoded)
        real_labels = torch.ones_like(logits_fake, device=logits_fake.device)
        g_loss = self.loss.adversarial_loss(logits_fake, real_labels)
        self.log('val_g_loss', g_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        l1_loss = self.loss.l1_loss(hr, decoded)
        self.log('val_l1_loss', l1_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        perceptual_loss = torch.mean(self.loss.perceptual_loss(hr, decoded))  
        self.log('val_perceptual_loss', perceptual_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)


    def configure_optimizers(self):
        disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.discriminator.lr, betas=(0.5, 0.999)) 
        vae_opt = torch.optim.Adam(self.vae.decoder.parameters(), lr=self.vae.lr, betas=(0.5, 0.999)) 
        return [vae_opt, disc_opt]
    
    def predict(self, x):
      sr = self.sampler.sample(x, self.unet, self.vae)
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




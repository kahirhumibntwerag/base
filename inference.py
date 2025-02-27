import torch.nn as nn
import torch
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
import numpy as np
import lightning as L
from src.metrics import kld_loss
import yaml
import argparse
from omegaconf import OmegaConf
from src.rrdb.RRDB import LightningGenerator

class Predictor:
    def __init__(self):
        pass
    @torch.no_grad()
    def predict(self, x: torch.Tensor, model,batch_size=4, transform=None, inverse_transform=None, device='cuda') -> torch.Tensor:
        """Generate super-resolved image from input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, 1, W, H)
            batch_size (int): Size of batches to process
            transform (callable): Transform to apply to input tensor
            inverse_transform (callable): Transform to apply to output tensor
            device (torch.device): Device to run model on
            
        Returns:
            torch.Tensor: Super-resolved output tensor
        """
        self.model = model.eval().to(device)
        self.images = [] # Reset images list
        
        n = x.shape[0]
        for i in range(0, n, batch_size):
            batch_end = min(i + batch_size, n)  # Handle last batch
            batch = x[i:batch_end]  # Use batch_end instead of i+batch_size
            
            if transform:
                batch = transform(batch)
            
            batch = batch.to(device)
            output = self.model.predict(batch) # Use direct call instead of predict
            
            if inverse_transform:
                output = inverse_transform(output)
            
            output = output.cpu()
            self.images.append(output)
            
        # Concatenate all batches along dimension 0 (batch dimension)
        result = torch.cat(self.images, dim=0)
        self.images = [] # Clear images list
        return result
    

class Metrics:
    def __init__(self):
        pass
    
    def calculate_psnr(self, hr: torch.Tensor, sr: torch.Tensor):
        """Calculate PSNR between HR and SR images.
        
        Args:
            hr (torch.Tensor): High resolution images of shape (N,1,W,H)
            sr (torch.Tensor): Super resolved images of shape (N,1,W,H)
            
        Returns:
            list: List of N float values representing PSNR for each image pair
        """
        hr = [hr[i].unsqueeze(0) for i in range(hr.shape[0])]
        sr = [sr[i].unsqueeze(0) for i in range(sr.shape[0])]
        psnr = list(map(peak_signal_noise_ratio, hr, sr))
        psnr = [value.item() for value in psnr]
        return psnr

    def calculate_ssim(self, hr: torch.Tensor, sr: torch.Tensor):
        """Calculate SSIM between HR and SR images.
        
        Args:
            hr (torch.Tensor): High resolution images of shape (N,1,W,H) 
            sr (torch.Tensor): Super resolved images of shape (N,1,W,H)
            
        Returns:
            list: List of N float values representing SSIM for each image pair
        """
        hr = [hr[i].unsqueeze(0) for i in range(hr.shape[0])]
        sr = [sr[i].unsqueeze(0) for i in range(sr.shape[0])]
        ssim = list(map(structural_similarity_index_measure, hr, sr))
        ssim = [value.item() for value in ssim]
        return ssim

    def calculate_kld(self, hr: torch.Tensor, sr: torch.Tensor):
        """Calculate KLD between HR and SR images.
        
        Args:
            hr (torch.Tensor): High resolution images of shape (N,1,W,H)
            sr (torch.Tensor): Super resolved images of shape (N,1,W,H)
            
        Returns:
            list: List of N float values representing KLD for each image pair
        """
        hr = [hr[i].unsqueeze(0) for i in range(hr.shape[0])]
        sr = [sr[i].unsqueeze(0) for i in range(sr.shape[0])]
        kld = list(map(kld_loss, hr, sr))
        return kld
    
    def calculate_metrics(self, hr: torch.Tensor, sr: torch.Tensor):
        """Calculate PSNR, SSIM, and KLD between HR and SR images.
        
        Args:
            hr (torch.Tensor): High resolution images of shape (N,1,W,H)
            sr (torch.Tensor): Super resolved images of shape (N,1,W,H)
            
        Returns:
            dict: Dictionary containing lists of metrics for each image pair
                  with keys 'psnr', 'ssim', and 'kld'
        """
        psnr = self.calculate_psnr(hr, sr)
        ssim = self.calculate_ssim(hr, sr)
        kld = self.calculate_kld(hr, sr)
        return {
            'psnr': psnr,
            'ssim': ssim, 
            'kld': kld
        }, {
            'psnr': np.mean(psnr),
            'ssim': np.mean(ssim), 
            'kld': np.mean(kld)
        }

class Model:
    def __init__(self):
        self.models = ['rrdb', 'esrgan', 'ldm']
    
    def load_model(self, model_name,checkpoint_path):
        if model_name == 'rrdb':
            self.model = LightningGenerator.load_from_checkpoint(checkpoint_path)
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        return self.model
    
    def instantiate_model(self,config):

        if config.model_name not in self.models:
            raise ValueError(f"Invalid model name: {config.model_name}. Available models: {self.models}")
            
        if config.model_name == 'rrdb':
            self.model = LightningGenerator(config)
        else:
            raise ValueError(f"Model {config.model_name} is not yet implemented")
            
        return self.model

def rescalee(images):
    """Rescale tensor using log normalization"""
    images_clipped = torch.clamp(images, min=1)
    images_log = torch.log(images_clipped)
    max_value = torch.log(torch.tensor(20000))
    max_value = torch.clamp(max_value, min=1e-9)
    images_normalized = images_log / max_value
    return images_normalized

def inverse_rescalee(images_normalized):
    """Inverse rescale from normalized to original range"""
    max_value = torch.log(torch.tensor(20000.0))
    max_value = torch.clamp(max_value, min=1e-9)
    images_log = images_normalized * max_value
    images_clipped = torch.exp(images_log)
    return images_clipped

def load_model(model_name,checkpoint_path):
    """Load the trained model from checkpoint."""
    model = Model().load_model(model_name, checkpoint_path)
    model.eval()
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Test the model with optional config overrides.")
    
    # Required arguments
    parser.add_argument('--model_name', type=str, default='rrdb', help='the model name')
    parser.add_argument('--model_path', type=str, default='model.pt', help='Path to model checkpoint')
    parser.add_argument('--lr_path', type=str, default='lr.pt', help='Path to LR data')
    parser.add_argument('--hr_path', type=str, default='hr.pt', help='Path to HR data')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for prediction')
    parser.add_argument('--opt', nargs='+', default=None, help='Override config options')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    predictor = Predictor()
    metrics = Metrics()
    model = load_model(args.model_name, args.model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = torch.load(args.lr_path)
    hr = torch.load(args.hr_path)
    sr = predictor.predict(lr, model, batch_size=args.batch_size, device=device)

    metrics_dict, metrics_mean = metrics.calculate_metrics(hr, sr)
    print(metrics_dict)
    print(metrics_mean)

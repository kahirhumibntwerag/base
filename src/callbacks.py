import lightning as L
import torch
import wandb
import matplotlib.pyplot as plt
from PIL import Image
import io
import numpy as np
from typing import Any, Dict

class ImageLoggingCallback(L.Callback):
    """
    Lightning callback for logging images during training and validation.
    """
    def __init__(self, log_freq: int = 100):
        """
        Args:
            log_freq (int): Frequency of logging images (in batches)
        """
        super().__init__()
        self.log_freq = log_freq

    def _log_images(self, 
                   pl_module: L.LightningModule,
                   batch: Any,
                   batch_idx: int,
                   prefix: str = "train"):
        """
        Log images to wandb with proper error handling
        """
        if batch_idx % self.log_freq != 0:
            return

        try:
            lr, hr = batch
            with torch.no_grad():
                sr = pl_module.predict(lr)

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(sr[0].detach().cpu().numpy().squeeze(), cmap='afmhot')
            ax.axis('off')

            # Save figure to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            buf.seek(0)

            # Convert buffer to image
            image = Image.open(buf)
            image_np = np.array(image)

            # Log to wandb
            wandb_image = wandb.Image(
                image_np, 
                caption=f"{prefix}_image_batch_{batch_idx}"
            )
            pl_module.logger.experiment.log({
                f"{prefix}_image_afmhot_batch_{batch_idx}": wandb_image
            })

        except Exception as e:
            print(f"Error in image logging: {str(e)}")
            # Continue training even if visualization fails
            pass

    def on_train_batch_end(self, 
                          trainer: L.Trainer,
                          pl_module: L.LightningModule,
                          outputs: Dict,
                          batch: Any,
                          batch_idx: int) -> None:
        """Log images during training"""
        self._log_images(pl_module, batch, batch_idx, prefix="train")

    def on_validation_batch_end(self,
                              trainer: L.Trainer,
                              pl_module: L.LightningModule,
                              outputs: Dict,
                              batch: Any,
                              batch_idx: int) -> None:
        """Log images during validation"""
        self._log_images(pl_module, batch, batch_idx, prefix="val")
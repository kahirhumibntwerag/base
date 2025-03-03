import os
import torch
from lightning.pytorch.callbacks import Callback

class CheckpointLoaderCallback:
    def __init__(self, checkpoint_dir):
        """
        Initialize the CheckpointLoader callback.
        
        Args:
            checkpoint_dir (str): Directory where checkpoints are stored
        """
        super().__init__()
        self.checkpoint_dir = checkpoint_dir

    def _get_latest_checkpoint(self):
        """Find the latest checkpoint in the directory based on epoch number."""
        if not os.path.exists(self.checkpoint_dir):
            return None

        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pt')]
        if not checkpoints:
            return None

        # Sort checkpoints by epoch number
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return os.path.join(self.checkpoint_dir, checkpoints[-1])

    def on_fit_start(self, state):
        checkpoint_path = self._get_latest_checkpoint()
        if checkpoint_path is not None:
            print(f"Found checkpoint at {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path)
                state['model'].load_state_dict(checkpoint['model_state_dict'])
                state['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"Resumed training from epoch {checkpoint['epoch']}")
            except Exception as e:
                print(f"Error loading checkpoint: {str(e)}")
        else:
            print("No checkpoint found, starting training from scratch")


import os
import torch

class ModelSaverCallback:
    def __init__(self, save_dir, save_freq=5):
        """
        Initialize the ModelSaver callback.
        
        Args:
            save_dir (str): Directory where models will be saved
            save_freq (int): Save the model every n epochs
        """
        super().__init__()
        self.save_dir = save_dir
        self.save_freq = save_freq
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

    def on_train_epoch_end(self, state):
        if (state['current_epoch'] + 1) % self.save_freq == 0:
            filename = f'model_epoch_{state["current_epoch"] + 1}.pt'
            save_path = os.path.join(self.save_dir, filename)
            
            torch.save({
                'epoch': state['current_epoch'] + 1,
                'model_state_dict': state['model'].state_dict(),
                'optimizer_state_dict': state['optimizer'].state_dict(),
                'global_step': state['global_step']
            }, save_path)
            
            print(f"Model saved at epoch {state['current_epoch'] + 1} to {save_path}")

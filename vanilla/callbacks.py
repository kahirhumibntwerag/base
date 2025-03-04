import os
import torch
import wandb
import numpy as np

class CheckpointLoaderCallback:
    def __init__(self, checkpoint_path=None):
        """
        Initialize the CheckpointLoader callback.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file to load
        """
        self.checkpoint_path = checkpoint_path

    def on_fit_start(self, trainer, model):
        if self.checkpoint_path is not None and os.path.exists(self.checkpoint_path):
            print(f"Loading checkpoint from {self.checkpoint_path}")
            try:
                checkpoint = torch.load(self.checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.configure_optimizers().load_state_dict(checkpoint['optimizer_state_dict'])
                trainer.current_epoch = checkpoint['epoch']
                trainer.global_step = checkpoint['global_step']
                print(f"Successfully loaded checkpoint from epoch {checkpoint['epoch']}")
            except Exception as e:
                print(f"Error loading checkpoint: {str(e)}")
        else:
            print("No checkpoint provided or file not found, starting training from scratch")

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

    def on_train_epoch_end(self, trainer, model):
        if (trainer.current_epoch + 1) % self.save_freq == 0:
            filename = f'model_epoch_{trainer.current_epoch + 1}.pt'
            save_path = os.path.join(self.save_dir, filename)
            
            torch.save({
                'epoch': trainer.current_epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.configure_optimizers().state_dict(),
                'global_step': trainer.global_step
            }, save_path)
            
            print(f"Model saved at epoch {trainer.current_epoch + 1} to {save_path}")


class LoggerCallback:
    def __init__(self):
        """Initialize the Logger callback."""
        pass

    def on_validation_epoch_end(self, trainer, model):
        """Print all metrics at the end of each epoch"""
        epoch = trainer.current_epoch + 1
        metrics_str = [f"Epoch {epoch}:"]
        
        # Get all metrics from logged_metrics
        for metric_name, values in model.logged_metrics.items():
            if values:  # Check if we have any values
                avg_value = sum(values) / len(values)  # Calculate average for this epoch
                metrics_str.append(f"{metric_name} = {avg_value:.6f}")
        
        # Print all metrics on one line
        print(" | ".join(metrics_str))
        print("-" * 50)  # Separator line for readability
        
        # Clear metrics for next epoch
        model.logged_metrics = {k: [] for k in model.logged_metrics.keys()}

class WandbCallback:
    def __init__(self, project_name, run_name=None, config=None):
        """
        Initialize WandbCallback.
        
        Args:
            project_name (str): Name of the W&B project
            run_name (str, optional): Name of this specific run
            config (dict, optional): Configuration to log
        """
        self.project_name = project_name
        self.run_name = run_name
        self.config = config or {}
        self.initialized = False

    def on_fit_start(self, trainer, model):
        """Initialize W&B run at the start of training"""
        if not self.initialized:
            wandb.init(
                project=self.project_name,
                name=self.run_name,
                config=self.config,
                resume=True if self.run_name else False
            )
            self.initialized = True
            wandb.watch(model)

    def on_validation_epoch_end(self, trainer, model):
        """Log metrics at the end of each epoch"""
        # Calculate average for each metric and ensure scalar values
        metrics = {}
        for metric_name, values in model.logged_metrics.items():
            if values:  # Check if we have any values
                # Convert to numpy array for consistent handling
                values_array = np.array(values)
                metrics[metric_name] = float(np.mean(values_array))
        
        # Add epoch and step info
        metrics.update({
            'epoch': trainer.current_epoch + 1,
            'global_step': trainer.global_step
        })
        
        # Log to wandb
        wandb.log(metrics)
        
        # Clear metrics for next epoch
        model.logged_metrics = {k: [] for k in model.logged_metrics.keys()}

    def on_fit_end(self, trainer, model):
        """Cleanup wandb run"""
        if self.initialized:
            wandb.finish()

    def on_exception(self, trainer, model, exception):
        """Cleanup wandb run on exception"""
        if self.initialized:
            wandb.finish()

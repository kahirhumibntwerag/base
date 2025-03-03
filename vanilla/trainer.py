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

    def _run_callback_hook(self, hook_name):
        """Generic method to run callback hooks safely"""
        state = {
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'device': self.device,
            'max_epochs': self.max_epochs,
            'model': self.model,
            'optimizer': self.optimizer,
            'train_loader': self.train_loader,
            'val_loader': self.val_loader,
            'batch': self.batch,
            'batch_idx': self.batch_idx,
            'metrics': self.model.logged_metrics if hasattr(self, 'model') else None,
        }
        
        for callback in self.callbacks:
            if hasattr(callback, hook_name) and callable(getattr(callback, hook_name)):
                getattr(callback, hook_name)(state)

    def _train_epoch(self, model, train_loader):
        """Handle single training epoch"""
        self.on_train_epoch_start(model)
        model.train()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, leave=False)):
            self._train_step(model, batch, batch_idx)
        
        self.on_train_epoch_end(model, train_loader)

    def _train_step(self, model, batch, batch_idx):
        """Handle single training step"""
        self.batch = batch
        self.batch_idx = batch_idx
        self._run_callback_hook('on_train_batch_start')
        
        batch = batch.to(self.device)
        model.training_step(batch)
        self.global_step += 1
        
        self._run_callback_hook('on_train_batch_end')

    def _validation_epoch(self, model, val_loader):
        """Handle single validation epoch"""
        self.on_validation_epoch_start(model)
        model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, leave=False)):
                self._validation_step(model, batch, batch_idx)
        
        self.on_validation_epoch_end(model, val_loader)

    def _validation_step(self, model, batch, batch_idx):
        """Handle single validation step"""
        self._run_callback_hook('on_validation_batch_start')
        
        batch = batch.to(self.device)
        model.validation_step(batch)
        
        self._run_callback_hook('on_validation_batch_end')

    def _setup_training(self, model, datamodule):
        """Setup training environment"""
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        model.to(self.device)
        return train_loader, val_loader

    def fit(self, model, datamodule):
        """Main training loop with hooks"""
        try:
            # Store references
            self.model = model
            self.optimizer = model.configure_optimizers()
            self.train_loader = datamodule.train_dataloader()
            self.val_loader = datamodule.val_dataloader()
            
            self._run_callback_hook('on_fit_start')
            model.to(self.device)

            for epoch in trange(self.current_epoch, self.max_epochs):
                self.current_epoch = epoch
                
                # Training phase
                self._train_epoch(model, self.train_loader)
                
                # Validation phase
                self._validation_epoch(model, self.val_loader)

            self._run_callback_hook('on_fit_end')

        except Exception as e:
            self._run_callback_hook('on_exception')
            raise e

    # Hook methods
    def on_fit_start(self, model, datamodule):
        """Called when fit begins"""
        self._run_callback_hook('on_fit_start')

    def on_fit_end(self, model, datamodule):
        """Called when fit ends"""
        self._run_callback_hook('on_fit_end')

    def on_train_epoch_start(self, model):
        """Called when training epoch begins"""
        self._run_callback_hook('on_train_epoch_start')

    def on_train_batch_start(self, model, batch, batch_idx):
        """Called before training batch"""
        self._run_callback_hook('on_train_batch_start')

    def on_train_batch_end(self, model, batch, batch_idx):
        """Called after training batch"""
        self._run_callback_hook('on_train_batch_end')

    def on_train_epoch_end(self, model, train_loader):
        """Called when training epoch ends"""
        self._run_callback_hook('on_train_epoch_end')

    def on_validation_epoch_start(self, model):
        """Called when validation epoch begins"""
        self._run_callback_hook('on_validation_epoch_start')

    def on_validation_batch_start(self, model, batch, batch_idx):
        """Called before validation batch"""
        self._run_callback_hook('on_validation_batch_start')

    def on_validation_batch_end(self, model, batch, batch_idx):
        """Called after validation batch"""
        self._run_callback_hook('on_validation_batch_end')

    def on_validation_epoch_end(self, model, val_loader):
        """Called when validation epoch ends"""
        self._run_callback_hook('on_validation_epoch_end')

    def on_exception(self, model, exception):
        """Called when an exception occurs"""
        self._run_callback_hook('on_exception')




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


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
                getattr(callback, hook_name)(*args, **kwargs)

    def _train_epoch(self, model, train_loader):
        """Handle single training epoch"""
        self.on_train_epoch_start(model)
        model.train()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, leave=False)):
            self._train_step(model, batch, batch_idx)
        
        self.on_train_epoch_end(model, train_loader)

    def _train_step(self, model, batch, batch_idx):
        """Handle single training step"""
        self._run_callback_hook('on_train_batch_start', model, batch, batch_idx)
        
        batch = batch.to(self.device)
        model.training_step(batch)
        self.global_step += 1
        
        self._run_callback_hook('on_train_batch_end', model, batch, batch_idx)

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
        self._run_callback_hook('on_validation_batch_start', model, batch, batch_idx)
        
        batch = batch.to(self.device)
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
            self.on_fit_start(model, datamodule)
            train_loader, val_loader = self._setup_training(model, datamodule)

            for epoch in trange(self.current_epoch, self.max_epochs, leave=False):
                # Training phase
                self._train_epoch(model, train_loader)
                
                # Validation phase
                self._validation_epoch(model, val_loader)
                
                self.current_epoch += 1

            self.on_fit_end(model, datamodule)

        except Exception as e:
            self._run_callback_hook('on_exception', model, e)
            raise e

    # Hook methods
    def on_fit_start(self, model, datamodule):
        """Called when fit begins"""
        self._run_callback_hook('on_fit_start', 
                              model=model, 
                              datamodule=datamodule)

    def on_fit_end(self, model, datamodule):
        """Called when fit ends"""
        self._run_callback_hook('on_fit_end', 
                              model=model, 
                              datamodule=datamodule)

    def on_train_epoch_start(self, model):
        """Called when training epoch begins"""
        self._run_callback_hook('on_train_epoch_start', 
                              model=model, 
                              epoch=self.current_epoch)

    def on_train_batch_start(self, model, batch, batch_idx):
        """Called before training batch"""
        self._run_callback_hook('on_train_batch_start', 
                              model=model, 
                              batch=batch, 
                              batch_idx=batch_idx, 
                              global_step=self.global_step)

    def on_train_batch_end(self, model, batch, batch_idx):
        """Called after training batch"""
        metrics = model.logged_metrics  # Get the metrics logged during training step
        self._run_callback_hook('on_train_batch_end', 
                              model=model, 
                              batch=batch, 
                              batch_idx=batch_idx, 
                              metrics=metrics, 
                              global_step=self.global_step)

    def on_train_epoch_end(self, model, train_loader):
        """Called when training epoch ends"""
        metrics = model.logged_metrics
        self._run_callback_hook('on_train_epoch_end', 
                              model=model, 
                              metrics=metrics, 
                              epoch=self.current_epoch)

    def on_validation_epoch_start(self, model):
        """Called when validation epoch begins"""
        self._run_callback_hook('on_validation_epoch_start', 
                              model=model, 
                              epoch=self.current_epoch)

    def on_validation_batch_start(self, model, batch, batch_idx):
        """Called before validation batch"""
        self._run_callback_hook('on_validation_batch_start', 
                              model=model, 
                              batch=batch, 
                              batch_idx=batch_idx)

    def on_validation_batch_end(self, model, batch, batch_idx):
        """Called after validation batch"""
        metrics = model.logged_metrics
        self._run_callback_hook('on_validation_batch_end', 
                              model=model, 
                              batch=batch, 
                              batch_idx=batch_idx, 
                              metrics=metrics)

    def on_validation_epoch_end(self, model, val_loader):
        """Called when validation epoch ends"""
        metrics = model.logged_metrics
        self._run_callback_hook('on_validation_epoch_end', 
                              model=model, 
                              metrics=metrics, 
                              epoch=self.current_epoch)

    def on_exception(self, model, exception):
        """Called when an exception occurs"""
        self._run_callback_hook('on_exception', 
                              model=model, 
                              exception=exception)




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


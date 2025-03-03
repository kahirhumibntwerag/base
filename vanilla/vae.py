import torch
from tqdm import trange, tqdm
import torch.nn as nn
import wandb  # Import wandb
import matplotlib.pyplot as plt  # Add matplotlib import

class Trainer:
    def __init__(self, model, datamodule, checkpoint_path, project_name="super_resolution"):
        self.model = model
        self.checkpoint_path = checkpoint_path

        self.datamodule = datamodule
        self.datamodule.setup()
        self.train_loader = datamodule.train_dataloader()
        self.val_loader = datamodule.val_dataloader()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.train_losses = []
        self.val_losses = []
        self.start_epoch = 0
        self.save_interval = 10
        self.max_epochs = 100

        # Initialize wandb
        wandb.init(project=project_name, config={
            "learning_rate": 1e-4,
            "batch_size": len(next(iter(self.train_loader))[0]),  # Get batch size dynamically
            "max_epochs": self.max_epochs
        })
        wandb.watch(self.model, log="all")  # Log gradients and model parameters

    def train(self):
        if self.checkpoint_path:
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']

        for epoch in trange(self.start_epoch, self.max_epochs, leave=False):
            self.model.train()
            epoch_train_loss = 0  # Track mean loss

            for batch in tqdm(self.train_loader, leave=False):
                self.optimizer.zero_grad()
                lr, hr = batch
                lr, hr = lr.to(self.device), hr.to(self.device)
                sr = self.model(lr)
                loss = self.loss(sr, hr)

                self.train_losses.append(loss.item())
                epoch_train_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            epoch_train_loss /= len(self.train_loader)

            if epoch % self.save_interval == 0:
                self.save_checkpoint(epoch)

            # Validation loop
            self.model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(self.val_loader, leave=False)):
                    lr, hr = batch
                    lr, hr = lr.to(self.device), hr.to(self.device)
                    sr = self.model(lr)
                    loss = self.loss(sr, hr)
                    self.val_losses.append(loss.item())
                    epoch_val_loss += loss.item()

                    # Log images for the first batch
                    if batch_idx == 0:
                        # Convert tensors to numpy and select first image
                        sr_img = sr[0].cpu().numpy()
                        hr_img = hr[0].cpu().numpy()
                        
                        # Create figure for SR
                        plt.figure(figsize=(8, 8))
                        plt.imshow(sr_img.squeeze(), cmap='afmhot')
                        plt.axis('off')
                        sr_figure = plt.gcf()
                        plt.close()
                        
                        # Create figure for HR
                        plt.figure(figsize=(8, 8))
                        plt.imshow(hr_img.squeeze(), cmap='afmhot')
                        plt.axis('off')
                        hr_figure = plt.gcf()
                        plt.close()

                        # Log images to wandb
                        wandb.log({
                            "super_resolution": wandb.Image(sr_figure),
                            "high_resolution": wandb.Image(hr_figure)
                        }, step=epoch)

            epoch_val_loss /= len(self.val_loader)

            # Log losses to wandb
            wandb.log({
                "train_loss": epoch_train_loss,
                "val_loss": epoch_val_loss,
                "epoch": epoch
            })

            print(f"Epoch {epoch} mean train loss: {epoch_train_loss}")
            print(f"Epoch {epoch} mean val loss: {epoch_val_loss}")
            self.train_losses = []
            self.val_losses = []

    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        checkpoint_path = f"checkpoint_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
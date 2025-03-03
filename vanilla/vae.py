import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class Dataset(Dataset):
    def __init__(self, tensors, downsample_factor=1/4, transform=None):
        self.tensors = tensors
        self.downsample_factor = downsample_factor
        self.transform = transform

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        hr = self.tensors[idx]
        if len(hr.shape) == 2:
            hr = hr.unsqueeze(0)

        if self.transform:
            hr = self.transform(hr)

        hr = hr.float().view(-1, 1, 512, 512)
        lr = F.interpolate(
            hr,
            size=(int(512*self.downsample_factor), int(512*self.downsample_factor)),
            mode='bilinear',
            align_corners=False
        )

        return lr.squeeze(0), hr.squeeze(0)

class DataModule:
    def __init__(
        self,
        train_path: str = None,
        val_path: str = None,
        test_path: str = None,
        batch_size: int = 32,
        num_workers: int = 12,
        downsample_factor: float = 1/4,
        transform = None
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.downsample_factor = downsample_factor
        self.transform = transform

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_tensors = torch.load(self.train_path)
            val_tensors = torch.load(self.val_path)

            self.train_dataset = Dataset(
                tensors=train_tensors,
                downsample_factor=self.downsample_factor,
                transform=self.transform
            )

            self.val_dataset = Dataset(
                tensors=val_tensors,
                downsample_factor=self.downsample_factor,
                transform=self.transform
            )

        if stage == 'test' and self.test_path:
            test_tensors = torch.load(self.test_path)
            self.test_dataset = Dataset(
                tensors=test_tensors,
                downsample_factor=self.downsample_factor,
                transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        if self.test_path:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )



import torch
from tqdm.notebook import trange, tqdm
import torch.nn as nn
import wandb  # Import wandb
  # Import wandb
import matplotlib.pyplot as plt  # Add matplotlib import

class Trainer:
    def __init__(self, model, datamodule, checkpoint_path, project_name="super_resolution"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.checkpoint_path = checkpoint_path

        self.datamodule = datamodule
        self.datamodule.setup()
        self.train_loader = datamodule.train_dataloader()
        self.val_loader = datamodule.val_dataloader()

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


import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        identity = x  # Skip connection
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += identity  # Add residual
        return F.relu(out)

class Generator(nn.Module):
    def __init__(self, in_channels=1, initial_channel=64, num_res_blocks=4, upscale_factor=4, lr=1e-4):
        super(Generator, self).__init__()
        self.lr = lr  # Store learning rate

        self.initial_conv = nn.Conv2d(in_channels, initial_channel, kernel_size=3, padding=1)

        # Stack residual blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(initial_channel) for _ in range(num_res_blocks)])

        # Final convolution before upsampling
        self.final_conv = nn.Conv2d(initial_channel, initial_channel, kernel_size=3, padding=1)

        # Upsampling layer (PixelShuffle for super-resolution)
        self.upsample = nn.Sequential(
            nn.Conv2d(initial_channel, initial_channel * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(initial_channel, in_channels, kernel_size=3, padding=1)  # Output same channels as input
        )

    def forward(self, x):
        x = F.relu(self.initial_conv(x))
        x = self.res_blocks(x)
        x = self.final_conv(x)
        x = self.upsample(x)
        return x

# Instantiate the Generator
#generator = Generator(in_channels=1, initial_channel=64, num_res_blocks=4, upscale_factor=4, lr=1e-4)

#from src.rrdb.RRDB import Generator
model = Generator(in_channels=1,
                  initial_channel=64,
                  num_res_blocks=4,
                  upscale_factor=4,
                  lr=1e-4
                  )
datamodule = DataModule(
    train_path='drive/MyDrive/train.pt',
    val_path='drive/MyDrive/val.pt',
    test_path='drive/MyDrive/hr.pt',
    batch_size=8
    )
trainer = Trainer(model, datamodule, checkpoint_path=None)
trainer.train()

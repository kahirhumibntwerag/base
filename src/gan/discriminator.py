import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, channel_list=[16, 32, 64, 128], lr=1e-6):
        super(Discriminator, self).__init__()
        self.lr = lr
        
        # Build feature layers dynamically
        layers = []
        
        # Input layer
        layers.extend([
            nn.Conv2d(in_channels, channel_list[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        # Hidden layers
        for i in range(len(channel_list)-1):
            # First conv at current channel size
            layers.extend([
                nn.Conv2d(channel_list[i], channel_list[i], kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                
                # Second conv increasing channel size
                nn.Conv2d(channel_list[i], channel_list[i+1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        self.out = nn.Conv2d(channel_list[-1], 1, kernel_size=3, stride=1, padding=1)
            
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.out(x)
        return x

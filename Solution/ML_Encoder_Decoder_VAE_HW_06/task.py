import torch
from torch import nn
from torch.nn import functional as F


# Task 1
class Encoder(nn.Module):
    def __init__(self, img_size=128, latent_size=512, start_channels=16, downsamplings=5):
        super().__init__()
        self.Conv2D_1 = nn.Conv2d(3, start_channels, kernel_size=1)
        channels_in = start_channels
        channels_out = start_channels * 2
        self.Conv2D_2 = nn.ModuleList()
        for _ in range(downsamplings):
            self.Conv2D_2.append(nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1))
            self.Conv2D_2.append(nn.BatchNorm2d(channels_out))
            self.Conv2D_2.append(nn.ReLU())
            channels_in = channels_out
            channels_out *= 2
        self.Flatten = nn.Flatten()
        r = (img_size // (2 ** downsamplings))
        self.last_layer = nn.Sequential(
            nn.Linear(channels_in * (r**2), 256),
            nn.ReLU(),
            nn.Linear(256, 2 * latent_size)
        )

    def forward(self, x):
        x = self.Conv2D_1(x)
        for l in self.Conv2D_2:
            x = l(x)
        x = self.Flatten(x)
        x = self.last_layer(x)
        mu, var = x.chunk(2, dim=1)
        sigma = torch.exp(var * 0.5)
        z = mu + sigma * torch.randn_like(sigma)
        return z, (mu, sigma)



# Task 2

class Decoder(nn.Module):
    def __init__(self, img_size=128, latent_size=512, end_channels=16, upsamplings=5):
        super().__init__()
        size = img_size // (2 ** upsamplings)
        num_channels = end_channels * (2 ** upsamplings)
        self.first_layer = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_channels * (size ** 2))
        )
        self.unflatten = nn.Unflatten(1, (num_channels, size, size))
        channels_in = num_channels
        channels_out = num_channels // 2
        self.second_layer = nn.ModuleList()
        for _ in range(upsamplings):
            self.second_layer.append(nn.ConvTranspose2d(channels_in, channels_out, kernel_size=4, stride=2, padding=1))
            self.second_layer.append(nn.BatchNorm2d(channels_out))
            self.second_layer.append(nn.ReLU())
            channels_in = channels_out
            channels_out //= 2
        self.third_layer = nn.Conv2d(channels_in, 3, kernel_size=1, stride=1, padding=0)
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = self.first_layer(z)
        x = self.unflatten(x)
        for layer in self.second_layer:
            x = layer(x)
        x = self.third_layer(x)
        x = self.tanh(x)
        return x

# Task 3

class VAE(nn.Module):
    def __init__(self, img_size=128, downsamplings=3, latent_size=256, down_channels=6, up_channels=12):
        super().__init__()
        self.encoder = Encoder(img_size, latent_size, down_channels, downsamplings)
        self.decoder = Decoder(img_size, latent_size, up_channels, downsamplings)

    def forward(self, x):
        z, (mu, sigma) = self.encoder(x)
        x_pred = self.decoder(z)
        kld = 0.5 * (sigma**2 + mu**2 - (sigma**2).log() - 1)
        return x_pred, kld

    def encode(self, x):
        z, (mu, sigma) = self.encoder(x)
        return z

    def decode(self, z):
        x_pred = self.decoder(z)
        return x_pred

    def save(self):
        torch.save(self.state_dict(), __file__[:-7] + "model.pth")
    
    def load(self):
        self.load_state_dict(torch.load(__file__[:-7] + "model.pth", map_location=torch.device('cpu')))
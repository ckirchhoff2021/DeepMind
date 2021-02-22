import torch
import torch.nn as nn
import torch.nn.functional as F


class FcVAE(nn.Module):
    def __init__(self, input_dim=784, h_dim=400, z_dim=20):
        super(FcVAE, self).__init__()

        # encoder
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)  # mu
        self.fc3 = nn.Linear(h_dim, z_dim)  # log_sigma ** 2

        # decoder
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, input_dim)

    def forward(self, x):
        mu, log_sigma = self.encode(x)
        sampled_z = self.reparameterize(mu, log_sigma)
        res = self.decode(sampled_z)
        return res, mu, log_sigma

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc2(h)
        log_sigma = self.fc3(h)
        return mu, log_sigma

    def reparameterize(self, mu, log_sigma):
        std = torch.exp(log_sigma * 0.5)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, x):
        h = F.relu(self.fc4(x))
        res = torch.sigmoid(self.fc5(h))
        return res


class ConvVAE(nn.Module):
    def __init__(self, input_chns=1, z_dim=20):
        super(ConvVAE, self).__init__()
        self.z_dim = z_dim

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(input_chns, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Linear(6272, 1024)
        self.fc2 = nn.Linear(1024, self.z_dim)
        self.fc3 = nn.Linear(1024, self.z_dim)

        self.fc4 = nn.Linear(self.z_dim, 1024)
        self.fc5 = nn.Linear(1024, 6272)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h1 = self.encoder_conv(x)
        h1 = h1.view(h1.size(0), -1)
        h = F.relu(self.fc1(h1))
        mu = self.fc2(h)
        log_sigma = self.fc3(h)
        return mu, log_sigma

    def reparameterize(self, mu, log_sigma):
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h1 = F.relu(self.fc4(z))
        h2 = F.relu(self.fc5(h1))
        h = h2.view(h2.size(0), 128, 7, 7)
        res = self.decoder_conv(h)
        return res

    def forward(self, x):
        mu, log_sigma = self.encode(x)
        sampled_z = self.reparameterize(mu, log_sigma)
        res = self.decode(sampled_z)
        return res, mu, log_sigma


def vae_loss(outputs, inputs, mu, log_sigma):
    # rec_loss = F.mse_loss(outputs, inputs)
    rec_loss = F.binary_cross_entropy(outputs, inputs)
    kl_divergence = torch.mean(0.5 * torch.sum(mu ** 2 + torch.exp(log_sigma) - log_sigma - 1, 1))
    loss = rec_loss + divergence
    return loss, rec_loss, divergence


def main():
    net1 = FcVAE()
    x1 = torch.randn(2, 784)
    y1, mu, sigma = net1(x1)
    print(y1.size())
    print(mu.size())
    print(sigma.size())

    net2 = ConvVAE()
    x2 = torch.randn(1, 1, 28, 28)
    y2, mu, sigma = net2(x2)
    print(y2.size())
    print(mu.size())
    print(sigma.size())
    print(torch.sum(torch.exp(sigma)))


if __name__ == '__main__':
    main()



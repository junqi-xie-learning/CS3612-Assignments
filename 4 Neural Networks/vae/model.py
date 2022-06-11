import torch
import torch.nn as nn
import torch.nn.functional as F


def make_fc_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.ReLU(inplace=True),
    )


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = make_fc_block(784, 256)
        self.fc2_mu = nn.Linear(256, 32)
        self.fc2_var = nn.Linear(256, 32)
        self.fc3 = make_fc_block(32, 256)
        self.fc4 = nn.Linear(256, 784)
        self.output = nn.Sigmoid()

    def encode(self, x):
        x = self.fc1(x)
        mu = self.fc2_mu(x)
        logvar = self.fc2_var(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.fc3(z)
        z = self.fc4(z)
        output = self.output(z)
        return output

    def forward(self, x, interpolate=False, alpha=0.5):
        mu, logvar = self.encode(torch.flatten(x, 1))
        z = self.reparameterize(mu, logvar)
        if interpolate:
            z_interpolate = alpha * z[0] + (1 - alpha) * z[1]
            return self.decode(z_interpolate), mu, logvar
        else:
            return self.decode(z), mu, logvar


def loss_function(output, x, mu, logvar):
    reconstruction = F.binary_cross_entropy(output, torch.flatten(x, 1), reduction='sum')
    latent = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction + latent


def train(model, device, train_loader, optimizer):
    model.train()
    train_loss = 0
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, mu, logvar = model(data)
        loss = loss_function(output, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    train_loss /= len(train_loader.dataset)
    return train_loss


@torch.no_grad()
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    for data, _ in test_loader:
        data = data.to(device)
        output, mu, logvar = model(data)
        test_loss += loss_function(output, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    return test_loss

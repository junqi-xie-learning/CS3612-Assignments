import torch
import torch.nn as nn
import torch.nn.functional as F


def make_conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2)
    )


def make_fc_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.ReLU(inplace=True),
    )


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = make_conv_block(1, 64)
        self.conv2 = make_conv_block(64, 128)
        self.fc1 = make_fc_block(6272, 256)
        self.fc2 = nn.Linear(256, 10)
        self.output = nn.LogSoftmax(dim=1)

    def forward(self, x, visualize=False):
        features = []

        # Convolutional layers
        x = self.conv1(x)
        features.append(torch.flatten(x.detach(), 1))
        x = self.conv2(x)
        features.append(torch.flatten(x.detach(), 1))

        # Fully connected layers
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        features.append(x.detach())
        x = self.fc2(x)

        # Output layer
        output = self.output(x)
        features.append(output.detach())

        if visualize:
            return output, features
        else:
            return output


def train(model, device, train_loader, optimizer):
    model.train()
    train_loss = 0
    train_accuracy = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction='sum')
        loss.backward()
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        train_accuracy += pred.eq(target.view_as(pred)).sum().item()
        optimizer.step()

    train_loss /= len(train_loader.dataset)
    train_accuracy /= len(train_loader.dataset)
    return train_loss, train_accuracy


@torch.no_grad()
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    test_accuracy = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        test_accuracy += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy /= len(test_loader.dataset)
    return test_loss, test_accuracy

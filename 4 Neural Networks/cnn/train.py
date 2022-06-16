import torch
from torch import optim
from torchvision import datasets, transforms

import numpy as np
from matplotlib import pyplot as plt

from model import *


args = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'batch_size': 64,
    'test_batch_size': 1000,
    'epochs': 10,
    'lr': 1e-4
}

train_kwargs = {'batch_size': args['batch_size']}
test_kwargs = {'batch_size': args['test_batch_size']}
if torch.cuda.is_available():
    cuda_kwargs = {'num_workers': 0,
                   'pin_memory': True,
                   'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

train_set = datasets.MNIST('../datasets', train=True, transform=transforms.ToTensor(), download=True)
test_set = datasets.MNIST('../datasets', train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
test_loader = torch.utils.data.DataLoader(test_set, **test_kwargs)

model = ConvNet().to(args['device'])
optimizer = optim.Adam(model.parameters(), lr=args['lr'])

epochs = args['epochs']
losses, train_accuracies, test_accuracies = [], [], []
for epoch in range(1, epochs + 1):
    print(f'Epoch: {epoch}/{epochs}')

    train_loss, train_accuracy = train(model, args['device'], train_loader, optimizer)
    print('Train Loss: {:.6f}\tAccuracy: {:.4f}'.format(train_loss, train_accuracy))
    losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    test_loss, test_accuracy = test(model, args['device'], test_loader)
    print('Test Loss: {:.6f}\tAccuracy: {:.4f}'.format(test_loss, test_accuracy))
    test_accuracies.append(test_accuracy)

torch.save(model.state_dict(), '../models/mnist_cnn.pth')

plt.plot(np.array(range(1, epochs + 1)), losses)
plt.savefig('train/losses.pdf')
plt.show()

plt.plot(np.array(range(1, epochs + 1)), train_accuracies)
plt.savefig('train/train_accuracies.pdf')
plt.show()

plt.plot(np.array(range(1, epochs + 1)), test_accuracies)
plt.savefig('train/test_accuracies.pdf')
plt.show()

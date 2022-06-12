import torch
from torch import optim
from torchtext import datasets
from torchtext.data.functional import to_map_style_dataset

import numpy as np
from matplotlib import pyplot as plt

from preprocess import *
from model import *


args = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'batch_size': 64,
    'test_batch_size': 1000,
    'epochs': 10,
    'lr': 1e-4
}

train_kwargs = {'batch_size': args['batch_size'], 'collate_fn': collate_batch}
test_kwargs = {'batch_size': args['test_batch_size'], 'collate_fn': collate_batch}
if torch.cuda.is_available():
    cuda_kwargs = {'num_workers': 0,
                   'pin_memory': True,
                   'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

train_set, valid_set, _ = map(to_map_style_dataset, datasets.SST2('../datasets'))
train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
test_loader = torch.utils.data.DataLoader(valid_set, **test_kwargs)

model = LSTMModel(len(vocab)).to(args['device'])
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

torch.save(model.state_dict(), '../models/sst2_lstm.pth')

plt.plot(np.array(range(1, epochs + 1)), losses)
plt.savefig('train/losses.pdf')
plt.show()

plt.plot(np.array(range(1, epochs + 1)), train_accuracies)
plt.savefig('train/train_accuracies.pdf')
plt.show()

plt.plot(np.array(range(1, epochs + 1)), test_accuracies)
plt.savefig('train/test_accuracies.pdf')
plt.show()

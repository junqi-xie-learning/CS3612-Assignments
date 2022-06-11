import torch
from torchvision import datasets, transforms

import numpy as np
from matplotlib import pyplot as plt

from model import *


args = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'test_batch_size': 1000
}

test_kwargs = {'batch_size': args['test_batch_size']}
if torch.cuda.is_available():
    cuda_kwargs = {'num_workers': 0,
                   'pin_memory': True,
                   'shuffle': True}
    test_kwargs.update(cuda_kwargs)

test_set = datasets.FashionMNIST('../datasets', train=False, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, **test_kwargs)

model = VAE()
model.load_state_dict(torch.load('../models/fashionmnist_vae.pth'))

data, target = next(iter(test_loader))

for i, image in enumerate(data[:10]):
    output, _, _ = model(image)
    plt.imsave(f'reconstruction/{i}.png', output.view(28, 28).cpu().detach().numpy(), cmap='gray')

alphas = [0, 0.2, 0.4, 0.6, 0.8, 1]
for alpha in alphas:
    output, _, _ = model(data[:2], interpolate=True, alpha=alpha)
    plt.imsave(f'interpolation/{alpha}.png', output.view(28, 28).cpu().detach().numpy(), cmap='gray')

import torch
from torchvision import datasets, transforms

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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

test_set = datasets.MNIST('../datasets', train=False, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, **test_kwargs)

model = ConvNet()
model.load_state_dict(torch.load('../models/mnist_cnn.pth'))

data, _ = next(iter(test_loader))
output, features = model(data, visualize=True)
pred = output.argmax(dim=1).numpy()

for i, feature in enumerate(features):
    W = feature.detach().numpy()

    pca = PCA(n_components=2)
    W_pca = pca.fit_transform(W)
    plt.figure(figsize=(10, 8))
    for target in range(10):
        W_plot = W_pca[pred == target]
        plt.scatter(W_plot[:, 0], W_plot[:, 1], marker='o')
    plt.legend(range(10))
    plt.savefig(f'visualization/layer_{i + 1}_pca.pdf')
    plt.show()

    tsne = TSNE(n_components=2)
    W_tsne = tsne.fit_transform(W)
    plt.figure(figsize=(10, 8))
    for target in range(10):
        W_plot = W_tsne[pred == target]
        plt.scatter(W_plot[:, 0], W_plot[:, 1], marker='o')
    plt.legend(range(10))
    plt.savefig(f'visualization/layer_{i + 1}_tsne.pdf')
    plt.show()

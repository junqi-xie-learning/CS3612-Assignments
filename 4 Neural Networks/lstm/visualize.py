import torch
from torchtext import datasets
from torchtext.data.functional import to_map_style_dataset

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from preprocess import *
from model import *


args = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'test_batch_size': 1000
}

test_kwargs = {'batch_size': args['test_batch_size'], 'collate_fn': collate_batch}
if torch.cuda.is_available():
    cuda_kwargs = {'num_workers': 0,
                   'pin_memory': True,
                   'shuffle': True}
    test_kwargs.update(cuda_kwargs)

train_set, valid_set, _ = map(to_map_style_dataset, datasets.SST2('../datasets'))
test_loader = torch.utils.data.DataLoader(valid_set, **test_kwargs)

model = LSTMModel(len(vocab))
model.load_state_dict(torch.load('../models/sst2_lstm.pth'))

label, text, offsets = next(iter(test_loader))
output, feature = model(text, offsets, visualize=True)


W = feature.detach().numpy()

pca = PCA(n_components=2)
W_pca = pca.fit_transform(W)
plt.figure(figsize=(10, 8))
for target in range(2):
    W_plot = W_pca[output.argmax(1) == target]
    plt.scatter(W_plot[:, 0], W_plot[:, 1], marker='o')
plt.legend(range(2))
plt.savefig(f'visualization/layer_pca.pdf')
plt.show()

tsne = TSNE(n_components=2)
W_tsne = tsne.fit_transform(W)
plt.figure(figsize=(10, 8))
for target in range(2):
    W_plot = W_tsne[output.argmax(1) == target]
    plt.scatter(W_plot[:, 0], W_plot[:, 1], marker='o')
plt.legend(range(2))
plt.savefig(f'visualization/layer_tsne.pdf')
plt.show()

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_sz = input_size
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))
 
    def forward(self, x, init_states=None):
        seq_length = x.size()[0]
        hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(self.hidden_size).to(x.device) 
            c_t = torch.zeros(self.hidden_size).to(x.device)
        else:
            h_t, c_t = init_states
 
        for t in range(seq_length):
            x_t = x[t]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t = torch.sigmoid(gates[:self.hidden_size])  # input
            f_t = torch.sigmoid(gates[self.hidden_size:self.hidden_size * 2])  # forget
            g_t = torch.tanh(gates[self.hidden_size * 2:self.hidden_size * 3])
            o_t = torch.sigmoid(gates[self.hidden_size * 3:])  # output
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq)
        return hidden_seq, (h_t, c_t)


class LSTMModel(nn.Module):
    def __init__(self, vocab_size):
        super(LSTMModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, 64)
        self.lstm = LSTM(64, 64)
        self.fc = nn.Linear(64, 2)

    def forward(self, text, offsets, visualize=False):
        embedded = self.embedding(text, offsets)
        feature, _ = self.lstm(embedded)
        output = self.fc(feature)
        if visualize:
            return output, feature
        else:
            return output


def train(model, device, train_loader, optimizer):
    model.train()
    train_loss = 0
    train_accuracy, count = 0, 0
    for label, text, offsets in train_loader:
        label, text, offsets = label.to(device), text.to(device), offsets.to(device)
        optimizer.zero_grad()
        output = model(text, offsets)
        loss = F.cross_entropy(output, label, reduction='sum')
        loss.backward()
        train_loss += loss.item()
        train_accuracy += (output.argmax(1) == label).sum().item()
        count += label.size(0)
        nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
    
    train_loss /= len(train_loader.dataset)
    train_accuracy /= count
    return train_loss, train_accuracy


@torch.no_grad()
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    test_accuracy, count = 0, 0
    for label, text, offsets in test_loader:
        label, text, offsets = label.to(device), text.to(device), offsets.to(device)
        output = model(text, offsets)
        test_loss += F.cross_entropy(output, label, reduction='sum').item()
        test_accuracy += (output.argmax(1) == label).sum().item()
        count += label.size(0)
    
    test_loss /= len(test_loader.dataset)
    test_accuracy /= count
    return test_loss, test_accuracy

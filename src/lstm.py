import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from .data import Data
from .utils import (server_config, client_config, data_config)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
max_features = 101
maxlen = 127
embed_size = 64
hidden_size = 64
n_layers = 1
batch_size = 64

class LSTMModel(nn.Module):
    def __init__(self, feat_size, embed_size, hidden_size, n_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.feat_size = feat_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.num_classes = num_classes

        self.embedding = nn.Embedding(feat_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.activation = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)

    def forward(self, x, hidden):
        embedded_feats = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded_feats, hidden)
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out[:, -1, :]  # Use the last time step
        fc_out = self.fc(lstm_out)
        output = self.activation(fc_out)
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        h = (weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(device),
             weight.new(self.n_layers, batch_size, self.hidden_size).zero_().to(device))
        return h

# Padding function
def pad_sequences(encoded_domains, maxlen):
    domains = []
    for domain in encoded_domains:
        if len(domain) >= maxlen:
            domains.append(domain[:maxlen])
        else:
            domains.append([0] * (maxlen - len(domain)) + domain)
    return np.asarray(domains)

# Training function
def train(model, trainloader, criterion, optimizer, epoch):
    model.train()
    clip = 5
    h = model.init_hidden(batch_size)
    total_loss = 0

    for inputs, labels in tqdm(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.long()
        h = tuple([each.data for each in h])

        model.zero_grad()
        output, h = model(inputs, h)
        # Loss computation
        if model.num_classes == 1:
            loss = criterion(output.squeeze(), labels)  # Squeeze output if model has only 1 class (binary classification)
        else:
            loss = criterion(output, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(trainloader)

# Testing function
def test(model, testloader, criterion):
    model.eval()
    val_h = model.init_hidden(batch_size)
    eval_losses = []
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.long()
            val_h = tuple([each.data for each in val_h])
            output, val_h = model(inputs, val_h)

            eval_loss = criterion(output.squeeze() if model.num_classes == 1 else output, labels)
            eval_losses.append(eval_loss.item())

            if model.num_classes == 1:
                preds = (output.squeeze() >= 0.5).int()
            else:
                preds = output.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return np.mean(eval_losses), correct / total

start_model_lstm = LSTMModel(max_features, embed_size, hidden_size, n_layers, num_classes=(1 if data_config['num_classes'] == 2 else data_config['num_classes'])).to(device)

# Start training task
def start_training_task(trainloader, testloader):
    lr = 3e-6
    epochs = 1

    model = LSTMModel(max_features, embed_size, hidden_size, n_layers, num_classes=(1 if data_config['num_classes'] == 2 else data_config['num_classes'])).to(device)
    criterion = nn.BCELoss() if data_config['num_classes'] == 2 else nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(params=model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_loss = train(model, trainloader, criterion, optimizer, epoch)
        # eval_loss, accuracy = test(model, testloader, criterion)
        print(
            f"Epoch: {epoch + 1}/{epochs}",
            f"Training Loss: {train_loss:.4f}",
            # f"Eval Loss: {eval_loss:.4f}",
            # f"Accuracy: {accuracy:.4f}"
            f"Eval Loss: ",
            f"Accuracy: "
        )
    return model.state_dict()

# if __name__ == "__main__":
#     start_training_task()

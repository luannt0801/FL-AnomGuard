import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
from .logging_setting import logger
from .data import Data, get_dataloader
from .utils import (server_config, client_config, data_config)

device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")

class LeNet(nn.Module):
    
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.classifier(x)
        return x

class LeNet_5(nn.Module):
    
    def __init__(self, in_features=3, num_classes=10):
        super(LeNet, self).__init__()
        
        self.conv_block = nn.Sequential( nn.Conv2d(in_channels=in_features,
                                                   out_channels=6,
                                                   kernel_size=5,
                                                   stride=1),
                                         nn.Tanh(),
                                         nn.MaxPool2d(2,2),
                                         
                                         nn.Conv2d(in_channels=6,
                                                   out_channels=16,
                                                   kernel_size=5,
                                                   stride=1),
                                         nn.Tanh(),
                                         nn.MaxPool2d(2,2)
                                        )
        
        self.linear_block = nn.Sequential( nn.Linear(16*5*5, 120),
                                           nn.Tanh(),
                                           nn.Linear(120,84),
                                           nn.Tanh(),
                                           nn.Linear(84,10)
                                         )
        
    def forward(self, x):
        x = self.conv_block(x)
        x = torch.flatten(x,1)
        x = self.linear_block(x)
        return x

# def train(model, trainloader, critertion, optimizer):
#     model.train()

#     running_loss = 0
#     running_acc = 0

#     for inputs, labels in tqdm(trainloader, desc="Training", leave=False):
#         logger.debug(f"Batch {i}: inputs.shape={inputs.shape}, labels.shape={labels.shape}")
#         inputs = inputs.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()

#         outputs = model(inputs)
#         loss = critertion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         pred_labels = torch.argmax(outputs, dim=1)
        
#         running_loss += loss
#         running_acc += ((pred_labels == labels).sum().item()/len(labels)) 

#         del inputs, labels, outputs

#     train_loss = running_loss/len(trainloader)
#     train_acc = running_acc/len(trainloader)

#     return train_loss, train_acc

def trainning(model, trainloader, critertion, optimizer):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(tqdm(trainloader, desc="Training", leave=False)):
        logger.debug(f"Batch {i}: inputs.shape={inputs.shape}, labels.shape={labels.shape}")
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = critertion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    accuracy = 100. * correct / total
    return running_loss / len(trainloader), accuracy


def test(model, testloader, critertion, optimizer):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc="Testing", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = critertion(outputs, labels)
            _, pred_labels = torch.max(outputs, dim=1)

            total += labels.size(0)
            correct += (pred_labels == labels).sum().item()

    accuracy = 100*(correct/total)

    return accuracy

start_model_cnn = LeNet(num_classes=data_config['num_classes'])

def start_trainning_CNN(trainloader, testloader):
    logger.warning(f"Length of trainloader: {len(trainloader)}")
    logger.warning(f"start_trainning_CNN")
    model = LeNet(num_classes=data_config['num_classes'])
    epochs = 2
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    critertion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        train_loss, train_accuracy = trainning(model=model, trainloader=trainloader, critertion=critertion, optimizer=optimizer)
        # accuracy = test(model=model, testloader=testloader, critertion=critertion, optimizer=optimizer)

        # logger.info(f"\n Epoch: {epoch} | Trainning: Loss: {train_loss} - Acc: {train_accuracy} | Test: Acc: {accuracy}")
        logger.info(f"\n Epoch: {epoch} | Trainning: Loss: {train_loss} - Acc: {train_accuracy} | Test: Acc: ")
    logger.warning(f"done start_trainning_CNN")
    return model.state_dict()

# if __name__ == "__main__":
#     start_trainning_CNN()

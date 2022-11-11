# Imports
import numpy as numpy
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# run on gpu if possible:
device = 'cuda' if torch.cuda.is_available() else 'cpu'
""" device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
 """
#device = torch.device('cude' if torch.cuda.is_available() else 'cpu')


class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

        
class ResidualBlock(nn.Module):
    """
    The residual block used by ResNet.
    
    Args:
        in_channels: The number of channels (feature maps) of the incoming embedding
        out_channels: The number of channels after the first convolution
        stride: Stride size of the first convolution, used for downsampling
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()        
        if stride > 1 or in_channels != out_channels:
            # Add strides in the skip connection and zeros for the new channels.
            self.skip = Lambda(lambda x: F.pad(x[:, :, ::stride, ::stride],
                                               (0, 0, 0, 0, 0, out_channels - in_channels),
                                               mode="constant", value=0))
        else:
            self.skip = nn.Sequential()
            
        # TODO: Initialize the required layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, input):
        # TODO: Execute the required layers and functions
        x1 = F.relu(self.bn1(self.conv1(input)))
        x2 = self.bn2(self.conv2(x1))
        return F.relu(x2 + self.skip(input))


class ResidualStack(nn.Module):
    """
    A stack of residual blocks.
    
    Args:
        in_channels: The number of channels (feature maps) of the incoming embedding
        out_channels: The number of channels after the first layer
        stride: Stride size of the first layer, used for downsampling
        num_blocks: Number of residual blocks
    """
    
    def __init__(self, in_channels, out_channels, stride, num_blocks):
        super().__init__()
        
        # TODO: Initialize the required layers (blocks)
        blocks = [ResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(num_blocks - 1):
            blocks.append(ResidualBlock(out_channels, out_channels))
        self.blocks = nn.ModuleList(blocks)
        
    def forward(self, input):
        # TODO: Execute the layers (blocks)
        x = input
        for block in self.blocks:
            x = block(x)
        return x

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# Initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

##################################### Load Dataset ###
# train_dataset = datasets.MNIST(
#    root='data/', train = true, transform=transforms.ToTensor(), download=True)

dataset = datasets.FashionMNIST(root='data/', train=True,
                                transform=transforms.ToTensor(), download=True)
train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])

#train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_set = datasets.FashionMNIST(
    root='data/', train=False, transform=transforms.ToTensor(), download=True)
#test_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# Dataloaders
dataloaders = {}
dataloaders['train'] = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True, num_workers=0,
                                  pin_memory=True)
dataloaders['val'] = DataLoader(val_set, batch_size=batch_size,
                                shuffle=False, num_workers=0,
                                pin_memory=True)
dataloaders['test'] = DataLoader(test_set, batch_size=batch_size,
                                 shuffle=False, num_workers=0,
                                 pin_memory=True)


# Training of one epoch ### Helper fctn


def run_epoch(model, optimizer, criterion, dataloader, train):
    """
    Run one epoch of training or evaluation.

    Args:
        model: The model used for prediction
        optimizer: Optimization algorithm for the model
        dataloader: Dataloader providing the data to run our model on
        train: Whether this epoch is used for training or evaluation

    Returns:
        Loss and accuracy in this epoch.
    """
    # TODO: Change the necessary parts to work correctly during evaluation (train=False)

    device = next(model.parameters()).device

    # Set model to training mode (for e.g. batch normalization, dro2^pout)
    if train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    epoch_acc = 0.0

    # Iterate over data
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)

        # zero the parameter gradients
        if train:
            optimizer.zero_grad()

        # forward
        with torch.set_grad_enabled(train):
            pred = model(xb)
            loss = criterion(pred, yb)
            top1 = torch.argmax(pred, dim=1)
            ncorrect = torch.sum(top1 == yb)

            # backward + optimize only if in training phase
            if train:
                loss.backward()
                optimizer.step()

        # statistics
        epoch_loss += loss.item()
        epoch_acc += ncorrect.item()

    epoch_loss = epoch_loss / len(dataloader.dataset)
    epoch_acc /= len(dataloader.dataset)
    return epoch_loss, epoch_acc


########################### Training with early stopping ###

def fit(model, optimizer, lr_scheduler, dataloaders, max_epochs, patience):
    """
    Fit the given model on the dataset.

    Args:
        model: The model used for prediction
        optimizer: Optimization algorithm for the model
        lr_scheduler: Learning rate scheduler that improves training
                      in late epochs with learning rate decay
        dataloaders: Dataloaders for training and validation
        max_epochs: Maximum number of epochs for training
        patience: Number of epochs to wait with early stopping the
                  training if validation loss has decreased

    Returns:
        Loss and accuracy in this epoch.
    """

    best_acc = 0

    for epoch in range(max_epochs):
        train_loss, train_acc = run_epoch(
            model, optimizer, criterion, dataloaders['train'], train=True)
        lr_scheduler.step()
        print(
            f"Epoch {epoch + 1: >3}/{max_epochs}, train loss: {train_loss:.2e}, accuracy: {train_acc * 100:.2f}%")

        val_loss, val_acc = run_epoch(
            model, None, criterion, dataloaders['val'], train=False)
        print(
            f"Epoch {epoch + 1: >3}/{max_epochs}, val loss: {val_loss:.2e}, accuracy: {val_acc * 100:.2f}%")

        # TODO: Add early stopping and save the best weights (in best_model_weights)
        if val_acc >= best_acc:
            best_epoch = epoch
            best_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())

        # Early stopping
        if epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_model_weights)


#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
""" 
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4) """
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[100, 150], gamma=0.1)

# Fit model
fit(model, optimizer, lr_scheduler, dataloaders, max_epochs=50, patience=50)


""" 
# Train

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to gpu if possible
        data = data.to(device)
        targets = targets.to(device)

        scores = model(data)
        loss = criterion(scores, targets)

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # gradient descent/adam step
        optimizer.step()

# check accuracy on training and test set
 


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')"""

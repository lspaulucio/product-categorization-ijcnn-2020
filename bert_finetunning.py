# -*- coding: utf-8 -*-

from transformers import BertForSequenceClassification

import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim

from utils.dataset import MLBERT

# Defining seed for reproducibility
seed = 500
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Network Hyperparameters

n_classes = 1588
batch_size = 128
model_type = 'bert-base-multilingual-cased'
model = BertForSequenceClassification.from_pretrained(model_type, num_classes=n_classes)

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

if cuda:
    print("Cuda available")
    model.cuda()

kwargs = {'pin_memory': True} if cuda else {}

ml_train = MLBERT(train=True, file='local_train.pt')
ml_val = MLBERT(train=True, file='local_val.pt')

train_loader = torch.utils.data.DataLoader(ml_train, batch_size=batch_size, shuffle=True, **kwargs)
val_loader = torch.utils.data.DataLoader(ml_val, batch_size=batch_size, shuffle=False, **kwargs)

learning_rate = 1e-3
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

weights = torch.load('classes_weights.pt')
if cuda:
    weights = weights.cuda()

criterion = nn.CrossEntropyLoss(weight=weights)
n_epochs = 10
log_interval = 100
val_interval = 1

start_epoch = 0

for epoch in tqdm(range(start_epoch, n_epochs)):  # loop over the dataset multiple times
    running_loss = 0.0
    losses = []
    for i, data in tqdm(enumerate(train_loader, 0), desc="Training..."):
        model.train()
        # get the inputs; data is a list of [inputs, labels]
        sentence, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(sentence)
        loss = criterion(outputs[0], labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if (i+1) % log_interval == 0:
            print('[%d, %5d] loss: %.3f' % (epoch, i , running_loss / log_interval))
            losses.append(running_loss / log_interval)
            running_loss = 0.0

    if (epoch % val_interval) == 0:
        correct = 0
        total = 0
        y = np.array([])
        y_pred = np.array([])
        with torch.no_grad():
            model.eval()
            for data in tqdm(val_loader, desc="Validation..."):
                sentence, labels = data[0].to(device), data[1].to(device)
                _, outputs = model(sentence, labels=labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                y = np.hstack((y, labels.cpu()))
                y_pred = np.hstack((y_pred, predicted.cpu()))

        accuracy = (total, 100 * correct / total)
        balanced_accuracy = (total, 100 * balanced_accuracy_score(y, y_pred))
        print('Accuracy of the network on the %d validation sentences: %.3f %%' % accuracy)
        print('Balanced Accuracy Score on the %d validation sentences: %.3f %%' % balanced_accuracy)

        checkpoint = {'model_state_dict':model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'epochs': epoch,
                      'val_accuracy': accuracy,
                      'balanced_accuracy': balanced_accuracy,
                      'loss': losses}
        torch.save(checkpoint, 'model/checkpoints/model_{}epochs.pt'.format(epoch+1))

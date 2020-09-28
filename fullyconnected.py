# -*- coding: utf-8 -*-
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Subset, ConcatDataset
from utils.dataset import MLFeatures

from sklearn.model_selection import train_test_split

# Defining seed for reproducibility
seed = 500
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Network Hyperparameters
hidden_size = 768
num_classes = 1588
batch_size = 128

NUM_FOLDS = 10
n_epochs = 50
log_interval = 200
val_interval = 1

for id in range(NUM_FOLDS):

    model = nn.Linear(hidden_size, num_classes)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")

    if cuda:
        print("Cuda available")
        model.cuda()

    kwargs = {'pin_memory': True} if cuda else {}

    data = torch.load('model/checkpoints/metric_learning/folds/folds_KNN/processed/fold_{}_KNN_processed'.format(id))
    X = np.array(data['processed'].to_list())
    y = data['category'].to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y)

    train = MLFeatures(data=X_train, labels=y_train)
    val = MLFeatures(data=X_val, labels=y_val)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val, batch_size=128, shuffle=False, **kwargs)
    learning_rate = 1e-3
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    criterion = nn.CrossEntropyLoss()
 
    print("Processing Fold {}".format(id))

    best_acc = 0
    for epoch in tqdm(range(n_epochs)):  # loop over the dataset multiple times
        running_loss = 0.0
        losses = []
        for i, data in tqdm(enumerate(train_loader, 0), desc="Training..."):
            model.train()
            # get the inputs; data is a list of [inputs, labels]
            sentence, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # loss, outputs = model(sentence, labels=labels)
            outputs = model(sentence)
            loss = criterion(outputs, labels)
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
                    outputs = model(sentence)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    y = np.hstack((y, labels.cpu()))
                    y_pred = np.hstack((y_pred, predicted.cpu()))

            accuracy = (total, 100 * correct / total)
            balanced_accuracy = (total, 100 * balanced_accuracy_score(y, y_pred))
            print('Accuracy of the network on the %d validation sentences: %.3f %%' % accuracy)
            print('Balanced Accuracy Score on the %d validation sentences: %.3f %%' % balanced_accuracy)


        if balanced_accuracy[1] > best_acc:
            checkpoint = {'model_state_dict':model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'epochs': epoch,
                          'val_accuracy': accuracy,
                          'balanced_accuracy': balanced_accuracy,
                          'loss': losses}
            print("Best acc: {:.3f} | Bal. acc: {:.3f}".format(best_acc, balanced_accuracy[1]))
            best_acc = balanced_accuracy[1]
        else:
            print("Saving model. Best Acc: {:.3f}".format(best_acc))
            torch.save(checkpoint, 'results/fc_{}_KNN.pt'.format(id))
            break

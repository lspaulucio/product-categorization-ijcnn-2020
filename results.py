# coding: utf-8
# -*- coding: utf-8 -*-

from pytorch_transformers import BertForSequenceClassification

import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from utils.dataset import MLBERT

# Defining seed for reproducibility
seed = 500
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Network Hyperparameters

num_classes = 1588
batch_size = 128
model_type = 'bert-base-multilingual-cased'
model = BertForSequenceClassification.from_pretrained('model/')
checkpoint = torch.load('model/checkpoints/metric_learning/bert/model_5epochs.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

if cuda:
    print("Cuda available")
    model.cuda()

kwargs = {'pin_memory': True} if cuda else {}

ml_test = MLBERT(train=True, file='processed_data/local_test.pt')
val_loader = torch.utils.data.DataLoader(ml_test, batch_size=128, shuffle=False, **kwargs)

correct_k = [0 for _ in range(5)]
total = 0
y = np.array([])
y_pred = np.array([])
with torch.no_grad():
    model.eval()
    for data in tqdm(val_loader, desc="Validation..."):
        sentence, labels = data[0].to(device), data[1].to(device)
        _, outputs = model(sentence, labels=labels)
        _, p = torch.max(outputs.data, 1)
        y_pred = np.hstack((y_pred, p.cpu()))
        y = np.hstack((y, labels.cpu()))
        _, predicted = outputs.topk(10, 1, True, True)
        pred = predicted.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        for i, k in enumerate([1, 3, 5, 7, 10]):
            correct_k[i] += correct[:k].view(-1).float().sum(0, keepdim=True).cpu().numpy()
            

accuracy_score(y,y_pred)
balanced_accuracy_score(y,y_pred)
correct_k
for c in correct_k:
    print(c/len(ml_test))
    

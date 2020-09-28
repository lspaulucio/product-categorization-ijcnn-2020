# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
from joblib import dump
from utils.models import Ensemble
from utils.dataset import MLFeatures
from sklearn.metrics import balanced_accuracy_score

# Network Hyperparameters
hidden_size = 768
num_classes = 1588
batch_size = 128

model = Ensemble(folder='results/ML/')

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

if cuda:
    print("Cuda available")
    model.cuda()

kwargs = {'pin_memory': True} if cuda else {}


data = torch.load('model/checkpoints/metric_learning/folds/ml_test_processed.pt')
X = np.array(data['processed'].to_list())
y = data['category'].to_numpy()

test = MLFeatures(data=X, labels=y)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, **kwargs)

print("Processing Fold {}".format(id))
y_pred = np.zeros((1, 1588))
with torch.no_grad():
    model.eval()
    for data in tqdm(test_loader, desc="Evaluation..."):
        sentence, labels = data[0].to(device), data[1].to(device)
        outputs = model(sentence)
        y_pred = np.vstack((y_pred, np.array(outputs.cpu())))

y_pred = y_pred[1:]
dump(y_pred, 'model/checkpoints/metric_learning/folds/ensemble_fc_ML.pt')
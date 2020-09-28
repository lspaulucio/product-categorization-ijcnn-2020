# -*- coding: utf-8 -*-

from pytorch_transformers import BertForSequenceClassification

import random
import numpy as np
from tqdm import tqdm
import pandas as pd
from joblib import dump
import torch
from utils.dataset import MLBERT

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

kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}

ml_test = MLBERT(train=False, file='processed_data/ml_test_bert.pt')

test_loader = torch.utils.data.DataLoader(ml_test, batch_size=batch_size, shuffle=False, **kwargs)

y_pred = np.zeros((1, 1588))
with torch.no_grad():
    model.eval()
    for data in tqdm(test_loader, desc="Evaluation..."):
        sentence, labels = data[0].to(device), data[1].to(device)
        outputs = model(sentence)[0]
        y_pred = np.vstack((y_pred, np.array(outputs.cpu())))

y_pred = y_pred[1:]
dump(y_pred, 'model/checkpoints/metric_learning/folds/ensemble_bert_ML.pt')
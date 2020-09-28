# -*- coding: utf-8 -*-

from transformers import BertForSequenceClassification

import random
import numpy as np
from tqdm import tqdm
import pandas as pd

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
checkpoint = torch.load('model_5epochs.pt')
model.load_state_dict(checkpoint['model_state_dict'])

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

if cuda:
    print("Cuda available")
    model.cuda()

kwargs = {'num_workers': 2, 'pin_memory': True} if cuda else {}

classes_map = torch.load('classes_map.pt')
reverse_category = dict([(value, key) for (key, value) in classes_map.items()])

ml_test = MLBERT(file='processed_data/ml_test_bert_processed.pt')
test_loader = torch.utils.data.DataLoader(ml_test, batch_size=batch_size, shuffle=False, **kwargs)

with torch.no_grad():
    out = pd.DataFrame(columns=['id', 'category'])
    model.eval()
    for i, data in enumerate(tqdm(test_loader)):
        sentence, labels = data[0].to(device), data[1]
        outputs = model(sentence)
        _, predicted = torch.max(outputs[0].data, 1)
        batch_data = [pd.Series([j, k], index=out.columns) for j, k in zip(labels.numpy(), predicted.cpu().numpy())] 
        out = out.append(batch_data, ignore_index=True)

out['category'] = out['category'].apply(lambda x: reverse_category[x])
out.to_csv('submition.csv', index=False)        

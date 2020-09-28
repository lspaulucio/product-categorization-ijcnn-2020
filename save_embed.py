# -*- coding: utf-8 -*-

import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

from utils.dataset import MLBERT
from utils.models import BertSequenceEmbedding

def remove_classifier(model_state):
    model_state.pop('classifier.bias')                                                                                                                                              
    model_state.pop('classifier.weight')

    return model_state

def extract_embeddings(dataloader, model, cuda=True):
    with torch.no_grad():
        model.eval()
        embeddings = []
        labels = []
        for data, target in tqdm(dataloader):
            if cuda:
                data = data.cuda()
            embed = model(data)
            embeddings.extend(embed.data.cpu().numpy())
            labels.extend(target.numpy())

    return embeddings, labels


# Defining seed for reproducibility
seed = 500
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', dest='checkpoint', required=True)
    parser.add_argument('--data', '-d', dest='data', required=True)
    args = parser.parse_args()

    # Hyperparameters
    batch_size = 32

    model = BertSequenceEmbedding(path='model/')
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    model_state = remove_classifier(checkpoint['model_state_dict'])
    
    model.load_state_dict(model_state)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")

    if cuda:
        print("Cuda available")
        model.cuda()

    kwargs = {'pin_memory': True} if cuda else {}

    print("Processing ml test")
    ml_data = MLBERT(train=True, file=args.data)
    ml_loader = torch.utils.data.DataLoader(ml_data, batch_size=batch_size, shuffle=False, **kwargs)
    embeddings, labels = extract_embeddings(ml_loader, model, cuda=cuda)
    df = pd.DataFrame(columns=['processed', 'category'], data={'processed':embeddings, 'category':labels})
    torch.save(df, 'model/checkpoints/metric_learning/folds/local_test_processed.pt')

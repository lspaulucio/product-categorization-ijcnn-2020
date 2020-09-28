# -*- coding: utf-8 -*-

from __future__ import print_function
import warnings
import sys

import torch
import numpy as np
import pandas as pd

import os
from os.path import join, isfile
from torch.utils.data import Dataset

class MLBERT(Dataset):

    def __init__(self, train=True, root_folder='data/mercadolibre/', file=None):
        self.root_folder = root_folder

        if file is None:
            print("File should be provided")
            sys.exit(1)

        path = join(self.root_folder, file)
        data = torch.load(path)

        self.labels = data['category'].to_numpy()
        self.data = torch.LongTensor(data['processed'].to_list())
        self.len = len(self.data)
        
    def __len__(self):
        'Denotes the total number of samples'
        return self.len            

    def __getitem__(self, index):
        'Generates one sample of data'
        x, y = self.data[index], self.labels[index]
        return torch.LongTensor(x), y

    def get_data(self):
        return self.data

    def get_labels(self):
        return self.labels

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        return fmt_str

##################################################################################################
class MLFeatures(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.len = len(self.data)
        
    def __len__(self):
        'Denotes the total number of samples'
        return self.len            

    def __getitem__(self, index):
        'Generates one sample of data'
        x, y = self.data[index], self.labels[index]
        return torch.Tensor(x), y

    def get_data(self):
        return self.data

    def get_labels(self):
        return self.labels

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        return fmt_str


class MLBERT_OUT(Dataset):
    def __init__(self, train=True, root_folder='data/mercadolibre/processed_data/'):
        self.train = train
        self.root_folder = root_folder
        if self.train:
            self.root_folder = join(self.root_folder, 'bert_outputs.pt')
            self.data = torch.load(self.root_folder)
        else:
            self.root_folder = join(self.root_folder, 'bert_data/test/')

        self.len = len(self.data)
        
    def __len__(self):
        'Denotes the total number of samples'
        return self.len            

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        data = self.data[index]
        x, y = data[0], data[1]
        return x[0], y[0]
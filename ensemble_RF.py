# -*- coding: utf-8 -*-

import torch 
import random 
import argparse
import numpy as np 
import pandas as pd 
from time import time 
from joblib import load, dump 
from sklearn.ensemble import RandomForestClassifier 
 

if __name__ == "__main__":

    seed = 500 
    random.seed(seed) 
    torch.manual_seed(seed) 
    np.random.seed(seed) 
    
    NUM_JOBS = 120 
    NUM_FOLDS = 10 
    NUM_TREES = 50 

    parser = argparse.ArgumentParser()

    parser.add_argument('--folder', '-f', dest='folder', required=True)
    parser.add_argument('--test_folder', '-tf', dest='test', required=True)
    args = parser.parse_args()

    folder = args.folder 
    test_f = args.test
    for i in range(NUM_FOLDS): 
        start = time() 
        print('Processing fold {}'.format(i)) 
        data = torch.load(folder+'fold_{}_processed.pt'.format(i)) 
        x = np.array(data['processed'].to_list()) 
        y = data['category'].to_numpy() 
        print('Training model...') 
        model = RandomForestClassifier(n_estimators=NUM_TREES, n_jobs=NUM_JOBS) 
        model.fit(x, y) 
        print('Time elapsed: {:.3f} '.format((time()-start)/60)) 
        print('Predicting...') 
        # Due to the memory limitations the local_test was evaluated by parts
        for j in range(40): 
            print("Split {}".format(j)) 
            t = np.load(test_f+'data_{}.npy'.format(j)) 
            y_pred = model.predict_proba(t) 
            dump(y_pred, 'outputs/data_{}_{}_KNN_out'.format(i, j)) 
            print('Time elapsed: {:.3f} '.format((time()-start)/60))
        
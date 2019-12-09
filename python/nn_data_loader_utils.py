import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
import re

from torch.autograd import Variable

class Dataset(data.Dataset):
    def __init__(self, dataset, path_data_pkl, overfit_train=False, train_size_for_overfit=100):
        super().__init__()
        self.dataset = dataset

        # Load data
        df = pd.read_pickle(path_data_pkl)
        df['label0'] = df['label'].apply(lambda x: int(x == 0))
        df['label1'] = df['label'].apply(lambda x: int(x == 1))
        
        # Take sample of train set if trying to overfit train dataset
        if dataset == 'Train' and overfit_train:
            df = df.sample(n=train_size_for_overfit, random_state=0)
            
        # Extract features and labels
        
        self.labels = df[['label0', 'label1']].copy()
        
        self.features = df.drop(['label', 'label0', 'label1'], axis=1).copy()
        
        
        
        self.index = list(df.index)
                 
        # Data size
        self.data_length = len(self.index)
        
        # Initiate weights
        neg_weight = 1 - np.mean(self.labels['label1'].values)
        self.weights = [neg_weight, 1 - neg_weight]
           

    def weighted_loss(self, prediction, target, use_gpu):
        weights_npy = np.array([self.weights[int(t[0])] for t in target.data])
        weights_tensor = torch.FloatTensor(weights_npy)
        if use_gpu:
            weights_tensor = weights_tensor.cuda()
        # loss = F.binary_cross_entropy_with_logits(prediction, target, weight=Variable(weights_tensor))
        loss = F.binary_cross_entropy_with_logits(prediction, target)
        
        return loss
        
      
    def __getitem__(self, i):
        
        # Load data
        index = self.index[i]
        label = np.array(self.labels.loc[index, :].values, dtype=np.double)
        features = np.array(self.features.loc[index, :].values,
                            dtype=np.double)
        
        # Convert to Tensor
        features_tensor = torch.FloatTensor(features)
        label_tensor = torch.FloatTensor(label)
        
        return features_tensor, label_tensor

    def __len__(self):
        return self.data_length

def load_data(data_folder, batch_size, overfit_train=False, train_size_for_overfit=10):
    
    train_dataset = Dataset('Train', path_data_pkl=data_folder + '/train.pkl', 
                            overfit_train=overfit_train, train_size_for_overfit=train_size_for_overfit)
    val_dataset = Dataset('Validation', path_data_pkl=data_folder + '/val.pkl')
    test_dataset = Dataset('Test', path_data_pkl=data_folder + '/test.pkl')

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    return train_loader, val_loader, test_loader

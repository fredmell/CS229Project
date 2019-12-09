import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
import torch

from sklearn import metrics
from torch.autograd import Variable

from nn_data_loader_utils import load_data

import re

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score



def run_model(model, loader, train=False, optimizer=None, use_gpu=False):
    preds = []
    labels = []

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.
    num_batches = 0

    for X, label in loader:
        
        if train:
            optimizer.zero_grad()

        if use_gpu:
            X = X.cuda()
            label = label.cuda()
            
        X = Variable(X)

        label = Variable(label)
        
        logit = model.forward(X)

        loss = loader.dataset.weighted_loss(label, logit, use_gpu)
        total_loss += loss.item()

        pred = torch.sigmoid(logit)
        pred_npy = pred.data.cpu().numpy()[0][0]
        label_npy = label.data.cpu().numpy()[0][0]

        preds.append(pred_npy)
        labels.append(label_npy)

        if train:
            loss.backward()
            optimizer.step()
        num_batches += 1

    avg_loss = total_loss / num_batches

    fpr, tpr, threshold = metrics.roc_curve(labels, preds)
    auc = metrics.auc(fpr, tpr)

    return avg_loss, auc, preds, labels


"""
def evaluate(split, model_path, use_gpu):
    
    model_name = re.findall(r'.*/.*_M[0-9]+_(.*?)_.*', str(model_path))[0]
    
    train_loader, valid_loader, test_loader = load_data(model_name, use_gpu=use_gpu)

    if model_name == "alexnet":
        model = MRN_alexnet()
    elif model_name == "resnet50":
        model = MRN_resnet50()
    elif model_name == "vgg16":
        model = MRN_vgg16()
    elif model_name == "densenet161":
        model = MRN_densenet161()
    elif model_name == "googlenet":
        model = MRN_googlenet()
    elif model_name == "inception_v3":
        model = MRN_inception_v3() 
    elif model_name == "squeezenet1_1":
        model = MRN_squeezenet1_1() 
    
    state_dict = torch.load(model_path, map_location=(None if use_gpu else 'cpu'))
    model.load_state_dict(state_dict)

    if use_gpu:
        model = model.cuda()

    if split == 'train':
        loader = train_loader
    elif split == 'valid':
        loader = valid_loader
    elif split == 'test':
        loader = test_loader
    else:
        raise ValueError("split must be 'train', 'valid', or 'test'")

    loss, auc, preds, labels = run_model(model, loader)
    preds = np.array(preds)
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0
    labels = np.array(labels)
    
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
  

    print(f'{split} loss: {loss:0.4f}')
    print(f'{split} AUC: {auc:0.4f}')
    print(f'{split} Acc: {acc:0.4f}')
    print(f'{split} precision: {precision:0.4f}')
    print(f'{split} recall: {recall:0.4f}')
    print(f'{split} f1: {f1:0.4f}')
    
    return preds, labels



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--gpu', action='store_true')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    evaluate(args.split, args.model_path, args.gpu)
    
"""    

import argparse
import json
import numpy as np
import os
import torch

from datetime import datetime
from pytz import timezone
from pathlib import Path
from sklearn import metrics

from nn_evaluate_utils import run_model
from nn_data_loader_utils import load_data
from nn_model_utils import *

from sklearn.metrics import accuracy_score


def train(rundir, epochs, learning_rate, weight_decay, max_patience, factor, threshold, use_gpu, model_name):
    
    # Load data
    train_loader, val_loader, test_loader = load_data(model_name, use_gpu=use_gpu)
    
    # Load model
    model = corresponding_model(model_name)

    # Set GPU or CPU 
    if use_gpu:
        model = model.cuda()

    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), 
                                 learning_rate, 
                                 weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           patience=max_patience, 
                                                           factor=factor, 
                                                           threshold=threshold)

    best_val_loss = float('inf')
    
    start_time = datetime.now()
    
    date = datetime.now().astimezone(timezone('US/Pacific'))
    date = str(date.strftime('%d_%b_%Y_H%H_M%M'))
    
    rst_file_name = rundir + '/' + date + '_' + model_name + '_Best_Results'
    final_file_name = rundir + '/' + date + '_' + model_name + '_Trained_Model'
    progress_file_name = rundir + '/' + date + '_' + model_name + '_Progress.txt'
    
    f = open(progress_file_name, 'w')
    
    txt = "==         PARAMETERS         =="
    size = len(txt)
    f.write("="*size + '\n')
    f.write(txt + '\n')
    f.write("="*size + '\n')
    f.write("epochs: {}\n".format(epochs))
    f.write("learning rate: {:.4e}\n".format(learning_rate))
    f.write("weight decay: {:.4e}\n".format(weight_decay))
    f.write("max patience: {}\n".format(max_patience))
    f.write("factor: {:.4e}\n".format(factor))
    f.write("threshold: {:.4e}\n".format(threshold))
    f.write("\n")
    f.write("\n")
    
    txt = "==          PROGRESS          =="
    size = len(txt)
    f.write("="*size + '\n')
    f.write(txt + '\n')
    f.write("="*size + '\n')
    
    for epoch in range(epochs):
        change = datetime.now() - start_time
        
        progress = 'Starting epoch {}/{} - time passed: {}'.format(epoch+1, epochs, str(change))
        print(progress)
        f.write(progress + '\n')
        
        train_loss, train_auc, preds, labels = run_model(model, train_loader, train=True, optimizer=optimizer)
        preds = np.array(preds)
        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0
        train_acc = accuracy_score(labels, preds)
        print(f'train loss: {train_loss:0.4f}')
        print(f'train AUC: {train_auc:0.4f}')
        print(f'train Accuracy: {train_acc:0.4f}')

        val_loss, val_auc, preds, labels = run_model(model, val_loader)
        preds = np.array(preds)
        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0
        labels = np.array(labels)
        val_acc = accuracy_score(labels, preds)
        print(f'valid loss: {val_loss:0.4f}')
        print(f'valid AUC: {val_auc:0.4f}')
        print(f'valid Accuracy: {val_acc:0.4f}')
        
        # Save progress
        progress = "train loss: {:0.4f}\n".format(train_loss)
        progress += "train AUC: {:0.4f}\n".format(train_auc)

        progress += "valid loss: {:0.4f}\n".format(val_loss)
        progress += "valid AUC: {:0.4f}\n".format(val_auc)
        progress += "valid Accuracy: {:0.4f}\n".format(val_acc)
        f.write(progress + "\n")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print('='*40)
            print(f'Model Saved at {rst_file_name}')
            print('='*40)
            torch.save(model.state_dict(), rst_file_name)

    # Save progress
    f.write("\n")
    txt = "==          RESULTS           =="
    size = len(txt)
    f.write("="*size + '\n')
    f.write(txt + '\n')
    f.write("="*size + '\n')
    f.write("Results saved at {}\n".format(rst_file_name))
    f.close()
    
    # Save final model
    torch.save(model.state_dict(), final_file_name)


    """
def overfit_train_data(rundir, epochs, learning_rate, weight_decay, max_patience, 
                       factor, threshold, use_gpu, model_name, train_size_for_overfit):
 """
def overfit_train_data(overfit_train_loader, model, epochs, learning_rate, weight_decay, max_patience, 
                       factor, threshold, use_gpu=False):

    # Set GPU or CPU 
    if use_gpu:
        model = model.cuda()

    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), 
                                 learning_rate, 
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           patience=max_patience, 
                                                           factor=factor, 
                                                           threshold=threshold)
    # Start training
    start_time = datetime.now()
    for epoch in range(epochs):
        change = datetime.now() - start_time
        progress = 'starting epoch {}. time passed: {}'.format(epoch+1, str(change))
        print(progress)
        
        train_loss, train_auc, preds, labels = run_model(model, 
                                                         overfit_train_loader,
                                                         train=True, 
                                                         optimizer=optimizer,
                                                         use_gpu=use_gpu)
        print(f'train loss: {train_loss:0.4f}')
        print(f'train AUC: {train_auc:0.4f}')
        preds = np.array(preds)
        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0
        labels = np.array(labels)
        train_acc = accuracy_score(labels, preds)
        print(f'train Accuracy: {train_acc:0.4f}')
        
        scheduler.step(train_loss)
        
        print()
        
        # Stop if overfit
        if abs(train_acc - 1.) < 1e-3:
            break
    
    if abs(train_acc - 1.) < 1e-3:
        txt = "==          OVERFIT: SUCCESS!          =="
    else:
        txt = "==          OVERFIT: FAILED!           =="
    size = len(txt)
    print("="*size)
    print(txt)
    print("="*size)


    
    
"""
def get_parser():
    models = {"mlp"}
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--rundir', type=str, required=True)
    parser.add_argument('--model', type=str, required=True, choices=models)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--overfit_train', action='store_true')
    parser.add_argument('--train_size', default=10, type=int)
    parser.add_argument('--learning_rate', default=1e-05, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--max_patience', default=5, type=int)
    parser.add_argument('--factor', default=0.3, type=float)
    parser.add_argument('--threshold', default=c, type=float)
    
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu:
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.rundir, exist_ok=True)
    
    with open(Path(args.rundir) / 'args.json', 'w') as out:
        json.dump(vars(args), out, indent=4)

    if args.overfit_train:
        overfit_train_data(args.rundir, args.epochs, args.learning_rate, args.weight_decay,
                           args.max_patience, args.factor, args.threshold, args.gpu, args.model, args.train_size)
    else:
        train(args.rundir, args.epochs, args.learning_rate, args.weight_decay,
              args.max_patience, args.factor, args.threshold, args.gpu, args.model)
              
"""

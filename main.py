import numpy as np
import pandas as pd
import torch as tch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Subset

from tqdm import tqdm

import json
from simpleAE import simpleAE
from customDataset import customDataset
import os
from sklearn.model_selection import KFold

def main():
    print('starting run')
    print('Torch Cuda is Available:\t{}'.format(tch.cuda.is_available()))
    device = tch.device("cuda" if tch.cuda.is_available() else "cpu")

    model_config = load_config()

    if not os.path.exists("output"):
        os.mkdir("output")

    model_name = model_config['name']
    loss_type = model_config['loss']
    img_dir = model_config['datapath']
    #Loss, Epochs, Batch-size
    num_epochs = model_config['num_epochs']
    batch_size = model_config['batch_size']
    use_kfold = model_config['use_kfold']
    learning_rate = model_config['learning_rate']
    sched_size = int(num_epochs//5)
    weight_decay = model_config['weight_decay']
    gamma = model_config['gamma']
    y_name = model_config['y_name']
    print(f"Running {model_name} with {loss_type} loss on dataset {img_dir}")

    # Update your dataset loading logic based on dataset_file
    transform = transforms.Compose([transforms.ToTensor()])

    # Initialize Custom Dataset
    # img_dir = 'data/'
    data_custom = customDataset(img_dir)
    if use_kfold:
        print('running with Kfold')
        k_folds = 5
        kfold = KFold(n_splits=k_folds, shuffle=True)
        
        for fold, (train_ids, test_ids) in enumerate(kfold.split(data_custom)):

            print(f"FOLD {fold}")
            print("------------------------------")

            #TODO Implement dataloader with train and test split.
            print('Input Dimensions: ', input_dim)
            train_subsampler = Subset(data_custom, train_ids)
            test_subsampler = Subset(data_custom, test_ids)
            if loss_type == "MSE":
                criterion = nn.MSELoss()
            elif loss_type == "BCE":
                criterion = nn.BCELoss() 
            #Data Loaders to handle the graphs we made earlier
            t_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True, num_workers=4)
            v_loader = DataLoader(test_subsampler, batch_size=batch_size, shuffle=True, num_workers=4)
            
            if model_name == "simpleAE":
                model = simpleAE(input_dim, model_name=model_name).to(device)
            elif model_name == "labbGNN":
                model = labbGNN(input_dim, model_name=model_name).to(device)
            
            
            optimizer = optim.AdamW(model.parameters()
                            , lr=learning_rate
                            , weight_decay= weight_decay
                            )  # Adjust learning rate as needed
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=sched_size, gamma=gamma)
            model, train_losses, val_losses = model.fitGNN(t_loader, v_loader, num_epochs, optimizer, criterion, scheduler)
            df = pd.DataFrame(train_losses, columns=['train_losses'])
            df['validation_losses'] = val_losses
            df.to_csv(f'output/{model_name}_fold{fold}.csv', sep=';')
            # model.plot_history(train_losses, val_losses, strarg=f'fold_{fold}')
            # model.print_eval(fold=fold)
            _ = model.make_embedding()
    else:
    # gmake = GraphMake('data/solubility.csv')
        print('running with no Kfold')
        # Assuming data_custom is your dataset
        n_train = int(len(data_custom) * 0.7)  # 70% of data for training
        n_test = len(data_custom) - n_train
        data_train, data_test = random_split(data_custom, [n_train, n_test])

        # Assuming your images have shape [6, 4240, 4240], to get the input dimension
        input_dim = 6
        print('Input Dimensions:', input_dim)

        if loss_type == "MSE":
            criterion = nn.MSELoss()
        elif loss_type == "BCE":
            criterion = nn.BCELoss()
        print(f'criterion {criterion}')
        t_loader = DataLoader(data_train, batch_size=1, shuffle=True)
        v_loader = DataLoader(data_test, batch_size=1, shuffle=True)
        print('loaders done')
        if model_name == "simpleAE":
            model = simpleAE(input_dim, model_name=model_name).to(device)
        elif model_name == "labbGNN":
            model = labbGNN(input_dim, model_name=model_name).to(device)
        print(f'model: {model_name}')
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=sched_size, gamma=gamma)
        print('fitting')
        train_losses, val_losses = model.fitAE(t_loader, v_loader, num_epochs, optimizer, criterion, scheduler)

        df = pd.DataFrame(train_losses, columns=['train_losses'])
        df['validation_losses'] = val_losses
        df.to_csv(f'output/{model_name}.csv', sep=';')
        # model.plot_history(train_losses, val_losses)
        # model.print_eval()
        _ = model.make_embedding()  

def load_config():
    if not os.path.exists("config.json"):
        create_default_config()
    with open("config.json", "r") as f:
        return json.load(f)
    
def create_default_config():
    default_config = {
        "name": "simpleAE",
        "loss": "MSE",
        "datapath": "/scratch/project_2008672/",
        "num_epochs": 500,
        "batch_size": 32,
        "use_kfold": False,
        "learning_rate": 0.001,
        "weight_decay": 0,
        "y_name": "self",
        "gamma": 1.0,
        "_Comment Key DO NOT USE": "Specify targets for training. Self for AE. set1 for AE+labels",
    }
    with open("config.json", "w") as f:
        json.dump(default_config, f)

    

if __name__ == "__main__":
    main()
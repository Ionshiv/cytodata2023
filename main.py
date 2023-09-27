import numpy as np
import pandas as pd
import torch as tch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.nn import GCNConv, GATConv, summary as gsummary, global_mean_pool, global_max_pool
from torch_geometric.data import Data#, DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.utils import dropout_adj
from tqdm import tqdm

# from rdkit import Chem
# from rdkit.Chem import AllChem
# from rdkit.Chem import Descriptors
# from rdkit.Chem import Lipinski
# from matplotlib import pyplot as plt
import json
from labbGNN import labbGNN
from simpleGNN import simpleGNN
from binGNN import binGNN
from graphmake import GraphMake
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
    dataset_file = model_config['dataset']
    #Loss, Epochs, Batch-size
    num_epochs = model_config['num_epochs']
    batch_size = model_config['batch_size']
    use_kfold = model_config['use_kfold']
    learning_rate = model_config['learning_rate']
    sched_size = int(num_epochs//5)
    weight_decay = model_config['weight_decay']
    gamma = model_config['gamma']
    y_name = model_config['y_name']
    print(f"Running {model_name} with {loss_type} loss on dataset {dataset_file}")

    # Update your dataset loading logic based on dataset_file
    gmake = GraphMake(f'data/{dataset_file}', y_name=y_name)
    data_pyg = gmake.getPyG()

    if use_kfold:
        k_folds = 5
        kfold = KFold(n_splits=k_folds, shuffle=True)
        
        for fold, (train_ids, test_ids) in enumerate(kfold.split(data_pyg)):
            print(f"FOLD {fold}")
            print("------------------------------")
            data_train = data_pyg.iloc[train_ids]
            data_train.reset_index(drop=True, inplace=True)
            data_test = data_pyg.iloc[test_ids]
            data_test.reset_index(drop=True, inplace=True)

            input_dim = data_train.iloc[5].x.size(1)
            print('Input Dimensions: ', input_dim)
            
            if loss_type == "MSE":
                criterion = nn.MSELoss()
            elif loss_type == "BCE":
                criterion = nn.BCELoss() 
            #Data Loaders to handle the graphs we made earlier
            t_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
            v_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True)
            
            if model_name == "simpleGNN":
                model = simpleGNN(input_dim, model_name=model_name).to(device)
            elif model_name == "labbGNN":
                model = labbGNN(input_dim, model_name=model_name).to(device)
            elif model_name == "binGNN":
                model = binGNN(input_dim, model_name=model_name).to(device)
            
            
            optimizer = optim.AdamW(model.parameters()
                            , lr=learning_rate
                            , weight_decay= weight_decay
                            )  # Adjust learning rate as needed
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=sched_size, gamma=gamma)
            model, train_losses, val_losses = model.fitGNN(t_loader, v_loader, num_epochs, optimizer, criterion, scheduler)
            df = pd.DataFrame(train_losses, columns=['train_losses'])
            df['validation_losses'] = val_losses
            df.to_csv(f'output/{model_name}_fold{fold}.csv', sep=';')
            model.plot_history(train_losses, val_losses, strarg=f'fold_{fold}')
            model.print_eval(fold=fold)
            
    else:
    # gmake = GraphMake('data/solubility.csv')
        
        n_train = int(len(data_pyg) * 0.7) # 70% of data for training and 30% for testing
        indices = np.arange(n_train)
        data_train = data_pyg[indices[:n_train]]
        data_train.reset_index(drop=True, inplace=True)
        data_test = data_pyg[~data_pyg.isin(data_train)]
        data_test.reset_index(drop=True, inplace=True)


        input_dim = data_train.iloc[5].x.size(1)
        print('Input Dimensions: ', input_dim)
        
        if loss_type == "MSE":
            criterion = nn.MSELoss()
        elif loss_type == "BCE":
            criterion = nn.BCELoss() 
        #Data Loaders to handle the graphs we made earlier
        t_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
        v_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True)
        
        if model_name == "simpleGNN":
            model = simpleGNN(input_dim, model_name=model_name).to(device)
        elif model_name == "labbGNN":
            model = labbGNN(input_dim, model_name=model_name).to(device)
        elif model_name == "binGNN":
            model = binGNN(input_dim, model_name=model_name).to(device)
        
        
        optimizer = optim.AdamW(model.parameters()
                        , lr=learning_rate
                        , weight_decay= weight_decay
                        )  # Adjust learning rate as needed
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=sched_size, gamma=gamma)
        model, train_losses, val_losses = model.fitGNN(t_loader, v_loader, num_epochs, optimizer, criterion, scheduler)
        df = pd.DataFrame(train_losses, columns=['train_losses'])
        df['validation_losses'] = val_losses
        df.to_csv(f'output/{model_name}.csv', sep=';')
        model.plot_history(train_losses, val_losses)
        model.print_eval()

def load_config():
    if not os.path.exists("config.json"):
        create_default_config()
    with open("config.json", "r") as f:
        return json.load(f)
    
def create_default_config():
    default_config = {
        "name": "simpleGNN",
        "loss": "MSE",
        "dataset": "A.csv",
        "num_epochs": 10000,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "gamma": 0.81,
        "y_name": "target"
    }
    with open("config.json", "w") as f:
        json.dump(default_config, f)

if __name__ == "__main__":
    main()
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

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from matplotlib import pyplot as plt
from labbGNN import labbGNN
from simpleGNN import simpleGNN
from graphmake import GraphMake

def main():
    print('starting run')
    print('Torch Cuda is Available:\t{}'.format(tch.cuda.is_available()))
    device = tch.device("cuda" if tch.cuda.is_available() else "cpu")

    gmake = GraphMake('data/solubility.csv')
    data_pyg = gmake.getPyG()
    n_train = int(len(data_pyg) * 0.7) # 70% of data for training and 30% for testing
    indices = np.arange(n_train)
    data_train = data_pyg[indices[:n_train]]
    data_train.reset_index(drop=True, inplace=True)
    data_test = data_pyg[~data_pyg.isin(data_train)]
    data_test.reset_index(drop=True, inplace=True)


    input_dim = data_train.iloc[5].x.size(1)
    print('Input Dimensions: ', input_dim)
    #Loss, Epochs, Batch-size
    num_epochs = 10000
    batch_size = 32
    sched_size = int(num_epochs//5)
    weight_decay = 1e-4
    gamma = 0.81
    criterion = nn.MSELoss()
    #Data Loaders to handle the graphs we made earlier
    t_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    v_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True)
    model = labbGNN(input_dim).to(device)
    optimizer = optim.AdamW(model.parameters()
                       , lr=0.001
               #        , weight_decay= weight_decay*15
                       )  # Adjust learning rate as needed
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=sched_size, gamma=gamma)
    model, train_losses, val_losses = model.fitGNN(t_loader, v_loader, num_epochs, optimizer, criterion, scheduler)
    df = pd.DataFrame(train_losses, columns=['train_losses'])
    df['validation_losses'] = val_losses
    df.to_csv('output.csv', sep=';')
    model.plot_history(train_losses, val_losses, 'simple GNN')
    model.print_eval()




if __name__ == "__main__":
    main()
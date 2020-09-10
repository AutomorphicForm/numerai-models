# -*- coding: utf-8 -*-


import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from loss_coteaching import loss_coteaching, loss_coteaching_original



class Net4(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(310, 150)
        self.fc2 = nn.Linear(150, 60)
        self.fc3 = nn.Linear(60, 1)
        self.dropout = nn.Dropout(p = 0.7)
        
    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return(x)


def adjust_learning_rate(optimizer, epoch, alpha_plan, beta1_plan):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1

def load_data():
    print('Beginning data download....')
    dataset1 = pd.read_csv('https://numerai-public-datasets.s3-us-west-2.amazonaws.com/latest_numerai_training_data.csv.xz')
    features = dataset1.columns[dataset1.columns.str.startswith('feature')]
    print('Finished the download.')
    return(dataset1, features)



    
if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the data and select the features
    dataset1, features = load_data()
    
    
    
    from torch.utils.data import Dataset, TensorDataset, DataLoader
    xtraint = torch.from_numpy(dataset1[features].to_numpy()).float()
    ytraint = torch.from_numpy(dataset1['target_kazutsugi'].to_numpy()).float()
    
    tensor_dataset1 = TensorDataset(xtraint, ytraint)
    dataloader1 = DataLoader(tensor_dataset1, batch_size = 8192)
    
    
    # Set the forget rate, which will change as epochs are passed
    forget_rate = 0.5
    
    num_gradual = 10 #can be 5, 10 ,15
    exponent = 2 # can be 0.5, 1, 2
    n_epoch = 50
    epoch_decay_start = 5
    learning_rate = 0.001
    
    
    rate_schedule = np.ones(n_epoch)*forget_rate
    rate_schedule[:num_gradual] = np.linspace(0, forget_rate**exponent, num_gradual)
    
    
    # Adjust learning rate and betas for Adam Optimizer
            
    mom1 = 0.9
    mom2 = 0.1
    alpha_plan = [learning_rate] * n_epoch
    beta1_plan = [mom1] * n_epoch
    for i in range(epoch_decay_start, n_epoch):
        alpha_plan[i] = float(n_epoch - i) / (n_epoch - epoch_decay_start) * learning_rate
        beta1_plan[i] = mom2
    
    net_model1 = Net4().to(device)
    net_model2 = Net4().to(device)
    
    optimizer1 = torch.optim.Adam(net_model1.parameters(), lr = 0.001)
    optimizer2 = torch.optim.Adam(net_model2.parameters(), lr = 0.001)
    
    for epoch in range(1, n_epoch):
        net_model1.train()
        net_model2.train()
        adjust_learning_rate(optimizer1, epoch, alpha_plan, beta1_plan)
        adjust_learning_rate(optimizer2, epoch, alpha_plan, beta1_plan)
        
        for count, (xset, yset) in enumerate(dataloader1):
            xbatch = xset.to(device)
            ybatch = yset.to(device)
            yhat1 = net_model1(xbatch)
            yhat2 = net_model2(xbatch)
            
            
            loss_1, loss_2 = loss_coteaching_original(yhat1.view(-1,), yhat2.view(-1,), ybatch, forget_rate)
            
            
            optimizer1.zero_grad()
            loss_1.backward()
            optimizer1.step()
            optimizer2.zero_grad()
            loss_2.backward()
            optimizer2.step()
            
            if count % 50 == 0:
                print(loss_1.item())
                print(loss_2.item())
                
                
    torch.save(net_model1, 'coteaching_model1.pth')
    torch.save(net_model2, 'coteaching_model2.pth')


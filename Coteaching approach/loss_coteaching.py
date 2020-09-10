# -*- coding: utf-8 -*-

import numpy as np
import torch.nn as nn

def loss_coteaching_original(y_1, y_2, target, forget_rate):
    """
    Changed to use MSELoss().
    """
    loss_1 = nn.MSELoss(reduction = 'none')(y_1, target)
    ind_1_sorted = np.argsort(loss_1.cpu().detach().view(-1,))
    loss_1_sorted = loss_1[ind_1_sorted]
    
    loss_2 = nn.MSELoss(reduction = 'none')(y_2, target)
    ind_2_sorted = np.argsort(loss_2.cpu().detach().view(-1,))
    loss_2_sorted = loss_2[ind_2_sorted]
    
    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))
    
    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    
    
    loss_1_update = nn.MSELoss()(y_1[ind_2_update], target[ind_2_update])
    loss_2_update = nn.MSELoss()(y_2[ind_1_update], target[ind_1_update])
    
    
    return loss_1_update, loss_2_update



def loss_coteaching(y_1, y_2, target, forget_rate):
    """
    Weighted mean squared error, with training examples chosen by the 
    coteaching algorithm weighted twice as heavily.
    """
    loss_1 = nn.MSELoss(reduction = 'none')(y_1, target)
    ind_1_sorted = np.argsort(loss_1.cpu().detach().view(-1,))
    loss_1_sorted = loss_1[ind_1_sorted]
    
    loss_2 = nn.MSELoss(reduction = 'none')(y_2, target)
    ind_2_sorted = np.argsort(loss_2.cpu().detach().view(-1,))
    loss_2_sorted = loss_2[ind_2_sorted]
    
    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))
    
    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    
    if ind_1_update.shape[0] == y_1.shape[0]:
        loss_1_update = nn.MSELoss()(y_1[ind_2_update], target[ind_2_update])
        loss_2_update = nn.MSELoss()(y_2[ind_1_update], target[ind_1_update])
    else:
        ind_1_remainder=ind_1_sorted[num_remember:]
        ind_2_remainder=ind_2_sorted[num_remember:]
        
        
        loss_1_update = (2*nn.MSELoss()(y_1[ind_2_update], target[ind_2_update]) 
                         + nn.MSELoss()(y_1[ind_2_remainder],target[ind_2_remainder]))
        loss_2_update = (2*nn.MSELoss()(y_2[ind_1_update], target[ind_1_update]) 
                         + nn.MSELoss()(y_2[ind_1_remainder],target[ind_1_remainder]))
    
    return loss_1_update, loss_2_update


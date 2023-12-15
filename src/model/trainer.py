import numpy as np
from matplotlib import pyplot as plt

from src.supervision.model import SEFS_S_Phase
import lightning as pl
import torch
import torch.nn.functional as F
import math

EPS = 1e-6


class STrainer(pl.LightningModule):
    def __init__(self,
                 x_mean, # column wise mean of the whole data
                 correlation_mat, # correlation matrix of the whole data
                 selection_prob,    # pre-selected selection probability
                 model_params,
                 trainer_params,
                 imp_feature_idx
                 ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = SEFS_S_Phase(
            model_params=model_params
        )
        
        self.trainer_params = trainer_params
        self.alpha_coef = self.trainer_params['alpha']  # alpha coef for gate vec loss
        self.beta_coef=self.trainer_params['beta'] # beta coef for controlling the number of feature selected.  
        self.l1_coef = self.trainer_params['l1_coef'] # l1 norm regul. coef
        self.optimizer_params = trainer_params['optimizer_params']
        self.imp_feature_idx = imp_feature_idx
        self.x_mean = self._check_input_type(x_mean)  #x_mean: mean of the whole data, computed beforehand
        self.R = self._check_input_type(correlation_mat) # correlation matrix of the whole data ,computed beforehand

        self.L = torch.linalg.cholesky(self.R+1e-6*torch.eye(self.R.shape[1])) # compute cholesky decomposition of correlation matrix beforehand
        self.train_loss_y, self.train_loss_total = 0., 0.
        
    def _check_input_type(self, x):
        # check if the input is a torch tensor, if not, convert it to torch tensor 
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).to(dtype=torch.float32)
        return x
    
    def _check_device(self, x):
        # check if the input is on the same device as the model
        if x.device != self.device:
            x = x.to(self.device)
        return x
    
    def Gaussian_CDF(self, x):
        return 0.5 * (1. + torch.erf(x / math.sqrt(2.)))
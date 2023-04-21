from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple

from os import path as pt
import pickle
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from lib.arfnn import SimpleGenerator
from lib.test_metrics import test_metrics
from lib.utils import to_numpy

import numpy as np 

@dataclass
class BaseConfig:
    seed: int = 0
    batch_size: int = 200
    device: str = 'cpu'
    p: int = 3
    q: int = 3
    hidden_dims: Tuple[int] = 3 * (50,)
    total_steps: int = 1000
    experiment_directory: str = ''

    def update(self,k,v):
        setattr(self,k,v)

def is_multivariate(x):
    return True if x.shape[-1] > 1 else False


def get_standard_test_metrics(x):
    test_metrics_list = [
        test_metrics['abs_metric'](x, reg=0.1),
        test_metrics['acf_id'](x, max_lag=2, reg=0.3),
    ]
    if is_multivariate(x):
        test_metrics_list.append(test_metrics['cross_correl'](x, reg=0.1))
    return test_metrics_list


class BaseAlgo:
    def __init__(self, base_config, x_real):
        self.base_config = base_config
        self.batch_size = base_config.batch_size
        self.hidden_dims = base_config.hidden_dims
        self.p, self.q = base_config.p, base_config.q
        self.total_steps = base_config.total_steps

        self.device = base_config.device
        self.experiment_directory =  base_config.experiment_directory
        self.x_real = x_real
        self.dim = self.latent_dim = x_real.shape[-1]
        self.is_multivariate = True if self.dim>1 else False
        
        self.training_loss = defaultdict(list)
        self.test_metrics_list = get_standard_test_metrics(x_real)

        self.G = SimpleGenerator(self.p * self.dim, self.dim, self.hidden_dims, self.latent_dim).to(self.device)
        
        self.best_score = 100
        self.best_list = {}

    def fit(self):
        if self.batch_size == -1:
            self.batch_size = self.x_real.shape[0]
        for i in tqdm(range(self.total_steps), ncols=80):  # sig_pred, x_past, x_real
            self.step()
            #start to track the best score after the first 100 epochs
            if i>100 and self.check_test_metric():
              self.save_model('best_score')
              self.idx_best = i
              #print('Current best score : %.3f'%self.best_score)
        #Print

    def step(self):
        raise NotImplementedError('Needs implementation by child.')

    def evaluate(self, x_fake):
        for test_metric in self.test_metrics_list:
            with torch.no_grad():
                test_metric(x_fake[:10000])
            self.training_loss[test_metric.name].append(
                to_numpy(test_metric.loss_componentwise)
            )
         
    def check_test_metric(self):
        #get comput the average as the score 
        score_list = dict()
        for i, test_metric in enumerate(self.test_metrics_list):
            score_list[test_metric.name]=self.training_loss[test_metric.name][-1].mean().astype(float)
        score = np.mean(np.fromiter(score_list.values(),dtype=float))
        #compare with best score
        if score < self.best_score:
          self.best_score = score
          self.best_list = score_list
          # return true if current score is better 
          return True
          #else return false
        else:
          return False
          
    def save_model(self,suffix):
        filename = pt.join(self.experiment_directory, 'G_weights_%s.torch'%suffix)
        with open(filename, 'wb') as f:
          pickle.dump(self.G.state_dict(), f)
        
    def plot_losses(self):
        fig, axes = plt.subplots(len(self.test_metrics_list) + 1, 1, figsize=(10, 8))
        algo = type(self).__name__
        if algo in ['RCGAN', 'TimeGAN', 'RCWGAN']:
            axes[0].plot(self.training_loss['G_loss'], label='Generator loss')

            if algo in ['RCGAN', 'TimeGAN']:
                axes[0].plot(self.training_loss['D_loss'], label='Discriminator loss')
            elif algo =='RCWGAN':
                axes[0].plot(self.training_loss['D_loss'], label='Critic loss')
                axes[0].plot(self.training_loss['reg'], label='GP')
            axes[0].plot(self.training_loss['G_grad'], label='G_grad_norm')

        elif algo == 'GMMN':
            axes[0].plot(self.training_loss['MMD'], label='MMD')

        elif algo in ['SigCWGAN',]:
            loss = self.training_loss['loss']
            axes[0].plot(loss, label='Sig-$W_1$ loss')
            axes[0].set_yscale('log')

        elif algo in ['MCGAN','SigMCGAN',]:
                axes[0].plot(self.training_loss['D_real'], label='D_real')
                axes[0].plot(self.training_loss['D_fake'], label='D_fake')
                axes[0].plot(self.training_loss['G_grad'], label='G_grad')
               
        else:
            raise NotImplementedError('Algo "%s" not implemented' % algo)

        axes[0].grid()
        axes[0].legend()
        for i, test_metric in enumerate(self.test_metrics_list):
            axes[i + 1].plot(self.training_loss[test_metric.name], label=test_metric.name)
            axes[i + 1].grid()
            axes[i + 1].legend()
            axes[i + 1].set_ylim(bottom=0.)
            if i + 1 == len(self.test_metrics_list):
                axes[i + 1].set_xlabel('Generator weight updates')

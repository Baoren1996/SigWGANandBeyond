import functools
from dataclasses import dataclass

import torch
from torch import autograd

from lib.algos.base import BaseAlgo
from lib.arfnn import ResFNN
from lib.utils import sample_indices, grad_norm
from lib.augmentations import SignatureConfig
import torch.nn as nn
import numpy as np

#To be removed 
@dataclass
class SigCWGANConfig:
    mc_size: int
    sig_config_future: SignatureConfig
    sig_config_past: SignatureConfig

    def compute_sig_past(self, x):
        return augment_path_and_compute_signatures(x, self.sig_config_past)

    def compute_sig_future(self, x):
        return augment_path_and_compute_signatures(x, self.sig_config_future)
        
class CGANTrainer(object):
    def __init__(
            self,
            G,
            D,
            G_optimizer,
            D_optimizer,
            p,
            q,
            gan_algo,
            reg_param: float = 10.
    ):
        self.G = G
        self.D = D
        self.G_optimizer = G_optimizer
        self.D_optimizer = D_optimizer

        self.p = p
        self.q = q

        self.gan_algo = gan_algo
        self.reg_param = reg_param

    def G_trainstep(self, x_fake, x_real):
        toggle_grad(self.G, True)
        self.G.train()
        self.G_optimizer.zero_grad()
        d_fake = self.D(x_fake)
        self.D.train()
        gloss = self.compute_loss(d_fake, 1)
        if self.gan_algo in ['TimeGAN']:
            gloss = gloss + torch.mean((x_fake - x_real) ** 2)
        gloss.backward()
        # compute the gradient of generator
        G_grad_norm =  grad_norm(self.G.parameters())
        self.G_optimizer.step()
        return gloss.item(), G_grad_norm.item()

    def D_trainstep(self, x_fake, x_real):
        toggle_grad(self.D, True)
        self.D.train()
        self.D_optimizer.zero_grad()

        # On real data
        x_real.requires_grad_()
        d_real = self.D(x_real)
        dloss_real = self.compute_loss(d_real, 1)

        # On fake data
        x_fake.requires_grad_()
        d_fake = self.D(x_fake)
        dloss_fake = self.compute_loss(d_fake, 0)

        # Compute regularizer on fake/real
        dloss = dloss_fake + dloss_real
        dloss.backward()

        if self.gan_algo == 'RCWGAN':
            reg = self.reg_param * self.wgan_gp_reg(x_real, x_fake)
            reg.backward()
        else:
            reg = torch.ones(1)
        # Step discriminator params
        self.D_optimizer.step()

        # Toggle gradient to False
        toggle_grad(self.D, False)
        return dloss_real.item(), dloss_fake.item(), reg.item()

    def compute_loss(self, d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)
        if self.gan_algo in ['RCGAN', 'TimeGAN','GARCH']:
            return torch.nn.functional.binary_cross_entropy_with_logits(d_out, targets)
        elif self.gan_algo == 'RCWGAN':
            return (2 * target - 1) * d_out.mean()
        elif self.gan_algo == 'MCGAN':
            return torch.pow(d_out-target,2).mean() 

    def wgan_gp_reg(self, x_real, x_fake, center=1.):
        batch_size = x_real.size(0)
        eps = torch.rand(batch_size, device=x_real.device).view(batch_size, 1, 1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = self.D(x_interp)
        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()
        return reg


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


class GAN(BaseAlgo):
    def __init__(self, base_config, gan_algo, x_real):
        super(GAN, self).__init__(base_config, x_real)
        self.D_steps_per_G_step = base_config.num_D_steps #4
        self.D = ResFNN(self.dim * (self.p + self.q), 1, self.hidden_dims, True).to(self.device)
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr= base_config.G_lr, betas=(base_config.G_beta1, base_config.G_beta2)) 
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr= base_config.D_lr, betas=(base_config.G_beta1, base_config.G_beta2))

        self.gan_algo = gan_algo
        self.trainer = CGANTrainer(                      # central object to tune the GAN
            G=self.G, D=self.D, G_optimizer=self.G_optimizer, D_optimizer=self.D_optimizer,
            gan_algo=gan_algo, p=self.p, q=self.q,
        )

    def step(self):
        for i in range(self.D_steps_per_G_step):
            # generate x_fake
            z = torch.randn(self.batch_size, self.q, self.latent_dim).to(self.device)
            indices = sample_indices(self.x_real.shape[0], self.batch_size)
            x_past = self.x_real[indices, :self.p].clone().to(self.device)
            with torch.no_grad():
                x_fake = self.G(z, x_past.clone())
                x_fake = torch.cat([x_past, x_fake], dim=1)
            D_loss_real, D_loss_fake, reg = self.trainer.D_trainstep(x_fake, self.x_real[indices].to(self.device))
            if i == 0:
                self.training_loss['D_loss_fake'].append(D_loss_fake)
                self.training_loss['D_loss_real'].append(D_loss_real)
                self.training_loss['RCWGAN_reg'].append(reg)
        # Generator step
        indices = sample_indices(self.x_real.shape[0], self.batch_size)
        x_past = self.x_real[indices, :self.p].clone().to(self.device)
        x_fake = self.G.sample(self.q, x_past)
        x_fake_past = torch.cat([x_past, x_fake], dim=1)
        G_loss, G_grad = self.trainer.G_trainstep(x_fake_past, self.x_real[indices].clone().to(self.device))
        self.training_loss['D_loss'].append(D_loss_fake + D_loss_real)
        self.training_loss['G_loss'].append(G_loss)
        self.training_loss['G_grad'].append(G_grad)
        self.evaluate(x_fake)


    
class RCGAN(GAN,):
    def __init__(self, base_config, x_real):
        super(RCGAN, self).__init__(base_config, 'RCGAN', x_real)


class TimeGAN(GAN, ):
    def __init__(self, base_config, x_real):
        super(TimeGAN, self).__init__(base_config, 'TimeGAN', x_real)


class RCWGAN(GAN, ):
    def __init__(self, base_config, x_real):
        super(RCWGAN, self).__init__(base_config, 'RCWGAN', x_real)

def sample_x_fake(G, q, mc_size, x_past):
    x_past_mc = x_past.repeat(mc_size, 1, 1).requires_grad_()
    x_fake_mc = G.sample(q, x_past_mc)
    return  x_fake_mc

class MCGANTrainer(CGANTrainer):
    '''
    A Monte-Carlo GAN trainer used to trainning MCGAN,
    D step:
        the same as in original GAN
    G step: 
    '''
    def __init__(
            self,
            G,
            D,
            G_optimizer,
            D_optimizer,
            p,
            q,
            mc_size,
            G_scheduler=None,
            D_scheduler=None,
            d_loss: str=None,
            reg_param: float = 10.,
            gan_algo:str='MCGAN',
            grad_reg: bool=False, 
    ):
        super(MCGANTrainer, self).__init__(G,D,G_optimizer,D_optimizer,p,q,gan_algo,reg_param)
        self.mc_size=mc_size
        self.grad_reg=grad_reg
        self.d_loss=d_loss
        self.G_scheduler=G_scheduler
        self.D_scheduler=D_scheduler
      
    def G_trainstep(self, x_fake_mc, x_real,clip_grad=False):
        toggle_grad(self.G, True)
        self.G.train()
        self.G_optimizer.zero_grad()
        
        #on fake data
        d_fake_mc = self.D(x_fake_mc)
        d_fake = d_fake_mc.reshape(self.mc_size,x_real.size(0), -1).mean(0)
        
        #on real data
        d_real = self.D(x_real)
        self.D.train()
        gloss = self.compute_loss(d_real, d_fake)#gloss = self.compute_loss(d_real, d_fake)

        gloss.backward()

        # clip the gradient or not
        G_grad_norm = torch.nn.utils.clip_grad_norm_(self.G.parameters(), 10) if clip_grad else grad_norm(self.G.parameters())

        #update step 
        self.G_optimizer.step()

        #use scheduler or not
        if self.G_scheduler is not None:
          self.G_scheduler.step()  #

        return d_fake.mean().item(),d_real.mean().item() ,G_grad_norm.item()
    
    def compute_loss(self, d_real,d_fake):
          return  torch.pow(d_real-d_fake,2).mean()



class MCGAN(BaseAlgo):
    '''
    Monte Carlo GAN training algorithm
    base_config: dict to store nn hyperparameters
    x_real:  real sample
    '''
    def __init__(self, base_config, x_real):
        super(MCGAN, self).__init__(base_config, x_real)
        
        #Size of Monte Carlo simulation 
        self.mc_size = base_config.mc_size

        #get the past and future sample 
        self.x_past = x_real[:, :self.p]
        self.x_future = x_real[:, self.p:]
        
        self.D_steps_per_G_step = base_config.num_D_steps 

        #Specify D network and optimizer 
        self.D = ResFNN(self.dim * (self.p + self.q), 1, self.hidden_dims, True).to(self.device)
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr= base_config.D_lr, betas=(base_config.D_beta1, base_config.D_beta2))  
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr= base_config.G_lr, betas=(base_config.G_beta1, base_config.G_beta2))
        
        #Load MCGAN trainder 
        self.trainer = MCGANTrainer(  
            G=self.G, D=self.D, G_optimizer=self.G_optimizer, D_optimizer=self.D_optimizer, p=self.p, q=self.q,mc_size=self.mc_size
        )

    def step(self):
        # Discriminator step
        for i in range(self.D_steps_per_G_step):
            # generate x_fake
            z = torch.randn(self.batch_size, self.q, self.latent_dim).to(self.device)
            # get the sampling indices of batch size
            indices = sample_indices(self.x_real.shape[0], self.batch_size)
            # get the minibatch of past paths
            x_past = self.x_real[indices, :self.p].clone().to(self.device)
            # Generate fake future path and concate it with past path
            with torch.no_grad():
                  x_fake = self.G(z, x_past.clone())
                  x_fake = torch.cat([x_past, x_fake], dim=1)
            # use real and fake samples for training D
            D_loss = self.trainer.D_trainstep(x_fake, self.x_real[indices].to(self.device))
            # Store the D loss
            if i == 0: 
                self.training_loss['D_loss'].append( D_loss)
               
        # Generator step
        indices = sample_indices(self.x_real.shape[0], self.batch_size)
        x_past = self.x_real[indices, :self.p].clone().to(self.device)
        # sampling mc_size number of  future paths for each past path 
        x_fake_mc = sample_x_fake(self.G, self.q, self.mc_size, x_past)
        # concatenate with past paths 
        x_fake_mc = torch.cat([x_past.repeat(self.mc_size,1,1), x_fake_mc], dim=1)
        # Training G,
        D_fake, D_real, grad_norm = self.trainer.G_trainstep(x_fake_mc, self.x_real[indices].clone().to(self.device))
        # Store the real and fake logits and G loss
        self.training_loss['D_real'].append(D_real)
        self.training_loss['D_fake'].append(D_fake)
        self.training_loss['G_grad'].append(grad_norm)
        #evaluate the model after each training epoch
        self.evaluate(x_fake_mc)
        



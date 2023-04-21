from dataclasses import dataclass

import math
import torch
from torch import autograd
from torch import optim
import torch.nn as nn
from torch.nn.utils import weight_norm

from tqdm import tqdm
from sklearn.linear_model import LinearRegression

from lib.arfnn import ResFNN,ResidualBlock
from lib.algos.base import BaseAlgo, BaseConfig
from lib.algos.gans import CGANTrainer
from lib.augmentations import SignatureConfig
from lib.augmentations import augment_path_and_compute_signatures
from lib.utils import sample_indices, to_numpy
from lib.algos.gans import MCGAN, MCGANTrainer, sample_x_fake



def sigcwgan_loss(sig_pred: torch.Tensor, sig_fake_conditional_expectation: torch.Tensor):
    return torch.norm(sig_pred - sig_fake_conditional_expectation, p=2, dim=1).mean()
    
def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


@dataclass
class SigCWGANConfig:
    mc_size: int
    sig_config_future: SignatureConfig
    sig_config_past: SignatureConfig

    def compute_sig_past(self, x):
        return augment_path_and_compute_signatures(x, self.sig_config_past)

    def compute_sig_future(self, x):
        return augment_path_and_compute_signatures(x, self.sig_config_future)
        
def calibrate_sigw1_metric(config, x_future, x_past):
    sigs_past = config.compute_sig_past(x_past)
    sigs_future = config.compute_sig_future(x_future)
    assert sigs_past.size(0) == sigs_future.size(0)
    X, Y = to_numpy(sigs_past), to_numpy(sigs_future)
    lm = LinearRegression()
    lm.fit(X, Y)
    sigs_pred = torch.from_numpy(lm.predict(X)).float().to(x_future.device)
    #plot_signature(sigs_past,sigs_future,sigs_pred)
    return sigs_pred
    
    
def plot_signature(sig_past,sig_future,sig_pred,alpha=0.3):
    import matplotlib.pyplot as plt
    _,ax=plt.subplots(2,1,figsize=(20,10))
    ax[0].plot(to_numpy(sig_past.mean(0)).T, label='past',alpha=alpha, linestyle='None', marker='o')
    ax[0].legend()
    ax[0].grid()
    ax[1].plot(to_numpy(sig_future.mean(0)).T,label='future', alpha=alpha, linestyle='None', marker='o')
    ax[1].plot(to_numpy(sig_pred.mean(0)).T,label='pred',alpha=alpha, linestyle='None', marker='o')
    ax[1].legend()
    ax[1].grid()
    plt.savefig('sig_vs_pred.pdf',dpi=100)
    
    

def sample_sig_fake(G, q, sig_config, x_past):
    x_past_mc = x_past.repeat(sig_config.mc_size, 1, 1).requires_grad_()
    x_fake = G.sample(q, x_past_mc)
    sigs_fake_future = sig_config.compute_sig_future(x_fake)
    sigs_fake_ce = sigs_fake_future.reshape(sig_config.mc_size, x_past.size(0), -1).mean(0)
    return sigs_fake_ce, x_fake

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
  
def sample_x_fake(G, q, mc_size, x_past):
    x_past_mc = x_past.repeat(mc_size, 1, 1).requires_grad_()
    x_fake_mc = G.sample(q, x_past_mc)

    return  x_fake_mc


class SigCWGAN(BaseAlgo):
    def __init__(
            self,
            base_config: BaseConfig,
            config: SigCWGANConfig,
            x_real: torch.Tensor,
    ):
        super(SigCWGAN, self).__init__(base_config, x_real)
        self.sig_config = config
        self.mc_size = config.mc_size

        self.x_past = x_real[:, :self.p]
        x_future = x_real[:, self.p:]
        self.sigs_pred = calibrate_sigw1_metric(config, x_future, self.x_past)

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=base_config.G_lr)
        self.G_scheduler = optim.lr_scheduler.StepLR(self.G_optimizer, step_size=base_config.opt_step_size, gamma=base_config.gamma)

    def sample_batch(self, ):
        random_indices = sample_indices(self.sigs_pred.shape[0], self.batch_size)  # sample indices
        # sample the least squares signature and the log-rtn condition
        sigs_pred = self.sigs_pred[random_indices].clone().to(self.device)
        x_past = self.x_past[random_indices].clone().to(self.device)
        return sigs_pred, x_past

    def step(self):
        self.G.train()
        self.G_optimizer.zero_grad()  # empty 'cache' of gradients
        sigs_pred, x_past = self.sample_batch() # x_fake future?
        sigs_fake_ce, x_fake = sample_sig_fake(self.G, self.q, self.sig_config, x_past)
        loss = sigcwgan_loss(sigs_pred, sigs_fake_ce)
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(self.G.parameters(), 10)#why 10
        self.training_loss['loss'].append(loss.item())
        self.training_loss['total_norm'].append(total_norm)
        self.G_optimizer.step()
        self.G_scheduler.step()  #
        self.evaluate(x_fake)

        

class SigCWGAN_G(SigCWGAN):
    '''
    A GARCH_based generator  
    '''
    def __init__(
            self,
            base_config: BaseConfig,
            config: SigCWGANConfig,
            x_real: torch.Tensor,
    ):
        super(SigCWGAN_G, self).__init__(base_config,config, x_real)
        self.G=GARCH_G(1,1,int(x_real.shape[-1]/2)).to(self.device)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=1e-3)
        self.G_scheduler = optim.lr_scheduler.StepLR(self.G_optimizer, step_size=100, gamma=0.9)


        
class SigMCGAN(MCGAN):
    def __init__(self, base_config, config: SigCWGANConfig, x_real):
        super(SigMCGAN, self).__init__(base_config,x_real)

        self.sig_config = config
        self.mc_size = config.mc_size
        config.sig_config_past.depth
        self.x_past = x_real[:, :self.p]
        self.x_future = x_real[:, self.p:]
        self.sig_future=self.sig_config.compute_sig_future(self.x_future)#compute the signature
        self.sig_dim=self.sig_future.shape[-1]
        
        self.D_steps_per_G_step = 3
        self.D = ResFNN(self.sig_dim, 1, self.hidden_dims, True).to(self.device)# D_Res(self.sig_dim, 1, self.hidden_dims,activation=nn.Identity()).to(self.device)
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr= base_config.G_lr, betas=(base_config.G_beta1, base_config.G_beta2))  # Using TTUR
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr= base_config.D_lr, betas=(base_config.D_beta1, base_config.D_beta2)) #lr very small
        self.G_scheduler = None#optim.lr_scheduler.StepLR(self.G_optimizer, step_size=100, gamma=0.9)
        self.D_scheduler = None#optim.lr_scheduler.StepLR(self.D_optimizer, step_size=100, gamma=0.9)
        self.mc_size = config.mc_size
        self.trainer =  MCGANTrainer(                                           # central object to tune the GAN
            G=self.G, D=self.D, G_optimizer=self.G_optimizer, D_optimizer=self.D_optimizer,p=self.p, q=self.q,mc_size=self.mc_size,G_scheduler=self.G_scheduler,D_scheduler=self.D_scheduler
        )
        
    def step(self):
        for i in range(self.D_steps_per_G_step):
            # generate x_fake
            z = torch.randn(self.batch_size, self.q, self.latent_dim).to(self.device)
            indices = sample_indices(self.x_real.shape[0], self.batch_size)
            x_past = self.x_past[indices].clone().to(self.device)
            with torch.no_grad():
                  x_fake = self.G(z, x_past.clone())
                  #x_fake = torch.cat(x_fake], dim=1)
                  sig_fake = self.sig_config.compute_sig_future(x_fake)
            D_loss = self.trainer.D_trainstep(sig_fake, self.sig_future[indices].to(self.device))
            if i == 0: 
                self.training_loss['D_loss'].append(D_loss)
        # Generator step
        indices = sample_indices(self.x_real.shape[0], self.batch_size)
        x_past = self.x_past[indices].clone().to(self.device)
        x_fake_mc = sample_x_fake(self.G, self.q, self.mc_size, x_past)
        #x_fake_mc = torch.cat([x_past.repeat(self.mc_size,1,1), x_fake_mc], dim=1)
        sig_fake_mc=self.sig_config.compute_sig_future(x_fake_mc)
        D_fake, D_real, grad_norm = self.trainer.G_trainstep(sig_fake_mc,self.sig_future[indices].clone().to(self.device),True)
        # Store the real and fake logits and G loss
        self.training_loss['D_real'].append(D_real)
        self.training_loss['D_fake'].append(D_fake)
        self.training_loss['G_grad'].append(grad_norm)
        self.evaluate(x_fake_mc)

class SigMCGAN_Cat(MCGAN):
    def __init__(self, base_config, config: SigCWGANConfig, x_real):
        super(SigMCGAN_Cat, self).__init__(base_config, config,x_real)

        self.sig_config = config
        self.mc_size = config.mc_size
        config.sig_config_past.depth
        self.x_past = x_real[:, :self.p]
        self.x_future = x_real[:, self.p:]
        self.sig_past = self.sig_config.compute_sig_past(self.x_past)#compute the signature
        self.sig_future = self.sig_config.compute_sig_future(self.x_future)#compute the signature
        self.sig_dim = self.sig_future.shape[-1] + self.sig_past.shape[-1]
        
        self.D_steps_per_G_step = 3
        self.D = ResFNN(self.sig_dim, 1, self.hidden_dims, True).to(self.device)# D_Res(self.sig_dim, 1, self.hidden_dims,activation=nn.Identity()).to(self.device)
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=1e-3, betas=(0, 0.9))  # Using TTUR
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=1e-3, betas=(0, 0.9)) #lr very small
        self.G_scheduler =  None #optim.lr_scheduler.StepLR(self.G_optimizer, step_size=100, gamma=0.9)
        self.D_scheduler = None #optim.lr_scheduler.StepLR(self.D_optimizer, step_size=100, gamma=0.9)
        self.mc_size = config.mc_size
        self.trainer =  MCGANTrainer(                                           # central object to tune the GAN
            G=self.G, D=self.D, G_optimizer=self.G_optimizer, D_optimizer=self.D_optimizer,p=self.p, q=self.q,mc_size=self.mc_size,G_scheduler=self.G_scheduler,D_scheduler=self.D_scheduler
        )
        
    def step(self):
        for i in range(self.D_steps_per_G_step):
            # generate x_fake
            z = torch.randn(self.batch_size, self.q, self.latent_dim).to(self.device)
            indices = sample_indices(self.x_real.shape[0], self.batch_size)
            x_past = self.x_past[indices].clone().to(self.device)
            sig_past = self.sig_past[indices].clone().to(self.device)
            sig_real = torch.cat([ sig_past, self.sig_future[indices].to(self.device)],dim=-1)
            with torch.no_grad():
                  x_fake = self.G(z, x_past.clone())
                  #x_fake = torch.cat(x_fake], dim=1)
                  sig_fake = self.sig_config.compute_sig_future(x_fake)
                  sig_fake = torch.cat([ sig_past, sig_fake],dim=-1)
            
            D_loss, reg = self.trainer.D_trainstep(sig_fake, sig_real)
            if i == 0: 
                self.training_loss['D_loss'].append(D_loss)
                self.training_loss['MCSWGAN_reg'].append(reg)
        # Generator step
        indices = sample_indices(self.x_real.shape[0], self.batch_size)
        x_past = self.x_past[indices].clone().to(self.device)
        sig_past = self.sig_past[indices].clone().to(self.device)
        sig_real = torch.cat([ sig_past, self.sig_future[indices].to(self.device)],dim=-1)
        x_fake_mc = sample_x_fake(self.G, self.q, self.mc_size, x_past)
        #x_fake_mc = torch.cat([x_past.repeat(self.mc_size,1,1), x_fake_mc], dim=1)
        sig_fake_mc = self.sig_config.compute_sig_future(x_fake_mc)
        sig_fake_mc = torch.cat([sig_past.repeat(self.mc_size,1), sig_fake_mc], dim=1)
        D_fake, D_real, G_loss = self.trainer.G_trainstep(sig_fake_mc,sig_real,True)
        self.training_loss['D_real'].append(D_real)
        self.training_loss['D_fake'].append(D_fake)
        self.training_loss['G_loss'].append(G_loss)
        self.evaluate(x_fake_mc)


class LinearBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        #self.activation = nn.PReLU()

    def forward(self, x):
        y = self.linear(x)#self.activation(self.linear(x)
        return y
        
class ProDiscriminator(nn.Module):
    def __init__(self, sig_past_dim,sig_future_dim,hidden_dims):
        super(ProDiscriminator, self).__init__()
        #self.resfnn=ResFNN(sig_future_dim,32,hidden_dims,True).to(self.device)
        self.linear1=nn.Linear(sig_future_dim,32)
        self.linear2=nn.Linear(32, 1)
        self.linear3=nn.Linear(sig_past_dim,32)
        self.activation=nn.ReLU()
        
    def forward(self,sig_future,sig_past):
        h=self.activation(self.linear1(sig_future))
        out=self.linear2(h)
        out=out+torch.sum(self.sigmoid(self.linear3(sig_past))*h,1,keepdim=True)
        return out

class ProDiscriminator(nn.Module):
    def __init__(self, sig_past_dim, sig_future_dim, hidden_dims):
        """
        Feedforward neural network with residual connection.
        Args:
            input_dim: integer, specifies input dimension of the neural network
            output_dim: integer, specifies output dimension of the neural network
            hidden_dims: list of integers, specifies the hidden dimensions of each layer.
                in above definition L = len(hidden_dims) since the last hidden layer is followed by an output layer
        """
        super(ProDiscriminator, self).__init__()
        blocks = list()

        input_dim_block =sig_future_dim
        for hidden_dim in hidden_dims:
            blocks.append(ResidualBlock(input_dim_block, hidden_dim))
            input_dim_block = hidden_dim
        self.linear = nn.Linear(input_dim_block, 1)
        self.linear1 = nn.Linear(sig_past_dim,input_dim_block)
        self.network = nn.Sequential(*blocks) 
        self.blocks = blocks
        self.sigmoid = nn.Sigmoid()
        self.relu=nn.ReLU()

    def forward(self, sig_future,sig_past):
        h = self.network(sig_future)
        #out=self.linear(h)
        out=torch.sum(self.sigmoid(self.linear1(sig_past))*self.relu(h),1,keepdim=True)
        return out
#

#out = self.linear(h) ############### The current output dim is 1
# Get projection of final featureset onto class vectors and add to evidence
#  out = out+ torch.sum(self.embed(y) * h, 1, keepdim=True) #
class ProjectionTrainer(CGANTrainer):
    '''
    A Monte-Carlo GAN trainer 
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
            gan_algo:str='MCWGAN_2',
            grad_reg: bool=False, 
            
    ):
        super(ProjectionTrainer, self).__init__(G,D,G_optimizer,D_optimizer,p,q,gan_algo,reg_param)
        self.mc_size=mc_size
        self.grad_reg=grad_reg
        self.d_loss=d_loss
        self.G_scheduler=G_scheduler
        self.D_scheduler=D_scheduler
        
    def G_trainstep(self, sig_past,sig_future, sig_fake,clip_grad=False):
        toggle_grad(self.G, True)
        self.G.train()
        self.G_optimizer.zero_grad()
        
        #on fake data
        d_fake= self.D(sig_fake,sig_past)
        #d_fake=d_fake_mc.reshape(self.mc_size,x_real.size(0), -1).mean(0)
        
        #on real data
        #d_real = self.D(sig_future,sig_past)
        self.D.train()
        gloss = -d_fake.mean()#torch.pow(d_fake-1,2).mean()

        gloss.backward()
        if clip_grad:
          total_norm = torch.nn.utils.clip_grad_norm_(self.G.parameters(), 10)
          #total_norm = torch.nn.utils.clip_grad_norm_(self.D.parameters(), 10)
        self.G_optimizer.step()
        if self.G_scheduler is not None:
          self.G_scheduler.step()  #
        return d_fake.mean().item(),gloss.item()
    
    def D_trainstep(self, sig_past,sig_future, sig_fake,):
        toggle_grad(self.D, True)
        self.D.train()
        self.D_optimizer.zero_grad()

        # On real data
        sig_future.requires_grad_()
        d_real = self.D(sig_future,sig_past)
        #dloss_real = self.compute_loss(d_real, 1)

        # On fake data
        sig_fake.requires_grad_()
        d_fake= self.D(sig_fake,sig_past)
        # Compute loss
        dloss_real=torch.pow(d_real-1,2).mean() 
        dloss_fake=torch.pow(d_fake,2).mean()
        dloss =dloss_real+dloss_fake#nn.ReLU()(1.0 - d_real).mean() + nn.ReLU()(1.0 + d_fake).mean()#
        
        reg = self.reg_param * self.wgan_gp_reg(x_real, x_fake) if self.grad_reg else 0
        dloss += reg
        dloss.backward()
 
        # Step discriminator params
        self.D_optimizer.step()
        if self.D_scheduler is not None:
          self.D_scheduler.step()  #

        # Toggle gradient to False
        toggle_grad(self.D, False)
        return dloss.item(),d_real.mean().item(), reg


class ProSigGAN(MCGAN):
    def __init__(self, base_config, config: SigCWGANConfig, x_real):
        super(ProSigGAN, self).__init__(base_config, config,x_real)

        self.sig_config = config
        self.mc_size = config.mc_size
        self.x_past = x_real[:, :self.p]
        self.x_future = x_real[:, self.p:]
        self.sig_future = self.sig_config.compute_sig_future(self.x_future)#compute the signature
        self.sig_past = self.sig_config.compute_sig_past(self.x_past)
        plot_signature(self.sig_past,self.sig_future,self.sig_past,alpha=0.3)
        self.output_dim = self.sig_future.shape[-1]#self.x_future.shape[2]*self.x_future.shape[1]#self.sig_future.shape[-1]
        self.input_dim = self.sig_past.shape[-1]#self.x_past.shape[2]*self.x_past.shape[1]#self.sig_past.shape[-1]
        self.D_steps_per_G_step = 4
        self.D = ProDiscriminator(self.input_dim,self.output_dim,self.hidden_dims).to(self.device)
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=1e-3, betas=(0, 0.9))  # Using TTUR
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=1e-3, betas=(0, 0.9)) #lr very small
        self.G_scheduler = None#optim.lr_scheduler.StepLR(self.G_optimizer, step_size=100, gamma=0.9)
        self.D_scheduler = None#optim.lr_scheduler.StepLR(self.D_optimizer, step_size=100, gamma=0.9)
        
        self.trainer =  ProjectionTrainer(G=self.G, D=self.D, G_optimizer=self.G_optimizer, D_optimizer=self.D_optimizer,p=self.p, q=self.q,mc_size=self.mc_size,G_scheduler=self.G_scheduler,D_scheduler=self.D_scheduler
        )

    def step(self):
        for i in range(self.D_steps_per_G_step):
            # generate x_fake
            z = torch.randn(self.batch_size, self.q, self.latent_dim).to(self.device)
            indices = sample_indices(self.x_real.shape[0], self.batch_size)
            x_past = self.x_past[indices].clone().to(self.device)
            x_future = self.x_future[indices].clone().to(self.device)
            sig_past = self.sig_past[indices].clone().to(self.device)
            with torch.no_grad():
                  x_fake = self.G(z, x_past.clone())
                  sig_fake = self.sig_config.compute_sig_future(x_fake)

            D_loss, D_real, reg = self.trainer.D_trainstep(sig_past, self.sig_future[indices].to(self.device),sig_fake)#(x_past.reshape(self.batch_size,-1), x_future.reshape(self.batch_size,-1),x_fake.reshape(self.batch_size,-1))
            if i == 0: 
                self.training_loss['D_loss'].append(D_loss)
                self.training_loss['MCSWGAN_reg'].append(reg)
        # Generator step
        indices = sample_indices(self.x_real.shape[0], self.batch_size)
        x_past = self.x_past[indices].clone().to(self.device)
        x_future = self.x_future[indices].clone().to(self.device)
        sig_past = self.sig_past[indices].clone().to(self.device)
        x_fake = self.G.sample(self.q, x_past)
        #x_fake_mc = torch.cat([x_past.repeat(self.mc_size,1,1), x_fake_mc], dim=1)
        sig_fake=self.sig_config.compute_sig_future(x_fake)
        D_fake, G_loss = self.trainer.G_trainstep(sig_past, self.sig_future[indices].to(self.device),sig_fake)
        self.training_loss['D_real'].append(D_real)
        self.training_loss['D_fake'].append(D_fake)
        self.training_loss['G_loss'].append(G_loss)
        self.evaluate(x_fake)

class ProjectionMCTrainer(ProjectionTrainer):
    '''
    A Monte-Carlo GAN trainer 
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
            gan_algo:str='MCWGAN_2',
            grad_reg: bool=False, 
            
    ):
        super(ProjectionMCTrainer, self).__init__(G,D,G_optimizer,D_optimizer,p,q,\
          mc_size,G_scheduler,D_scheduler,d_loss,reg_param,gan_algo,grad_reg)
        
    def G_trainstep(self, sig_past,sig_future, sig_fake_mc,clip_grad=False):
        toggle_grad(self.G, True)
        self.G.train()
        self.G_optimizer.zero_grad()
        
        #on fake data
        sig_past_mc= sig_past.repeat(self.mc_size, 1)
        d_fake_mc = self.D(sig_fake_mc,sig_past_mc)
        d_fake = d_fake_mc.reshape(self.mc_size,sig_future.size(0), -1).mean(0)
        
        #on real data
        d_real = self.D(sig_future,sig_past)
        self.D.train()
        gloss =  torch.pow(d_real-d_fake,2).mean()

        gloss.backward()
        if clip_grad:
          total_norm = torch.nn.utils.clip_grad_norm_(self.G.parameters(), 10)

        self.G_optimizer.step()
        if self.G_scheduler is not None:
          self.G_scheduler.step()  
        return d_fake.mean().item(),gloss.item()
        
def sample_x_fake(G, q, mc_size, x_past):
    x_past_mc = x_past.repeat(mc_size, 1, 1).requires_grad_()
    x_fake_mc = G.sample(q, x_past_mc)
    return  x_fake_mc

class ProSigMCGAN(ProSigGAN):
    def __init__(self, base_config, config: SigCWGANConfig, x_real):
        super(ProSigMCGAN, self).__init__(base_config, config,x_real)
        self.trainer =  ProjectionMCTrainer(G=self.G, D=self.D, G_optimizer=self.G_optimizer, D_optimizer=self.D_optimizer,p=self.p, q=self.q,mc_size=self.mc_size,G_scheduler=self.G_scheduler,D_scheduler=self.D_scheduler
        )

    def step(self):
        # Discriminator step
        for i in range(self.D_steps_per_G_step):
            # generate x_fake
            z = torch.randn(self.batch_size, self.q, self.latent_dim).to(self.device)
            indices = sample_indices(self.x_real.shape[0], self.batch_size)
            x_past = self.x_past[indices].clone().to(self.device)
            x_future = self.x_future[indices].clone().to(self.device)
            sig_past = self.sig_past[indices].clone().to(self.device)
            with torch.no_grad():
                  x_fake = self.G(z, x_past.clone())
                  sig_fake = self.sig_config.compute_sig_future(x_fake)

            D_loss, D_real, reg = self.trainer.D_trainstep(sig_past, self.sig_future[indices].to(self.device),sig_fake)#(x_past.reshape(self.batch_size,-1), x_future.reshape(self.batch_size,-1),x_fake.reshape(self.batch_size,-1))
            if i == 0: 
                self.training_loss['D_loss'].append(D_loss)
                self.training_loss['reg'].append(reg)
                
        # Generator step
        indices = sample_indices(self.x_real.shape[0], self.batch_size)
        x_past = self.x_past[indices].clone().to(self.device)
        x_fake_mc = sample_x_fake(self.G, self.q, self.mc_size, x_past)
        sig_fake_mc=self.sig_config.compute_sig_future(x_fake_mc)
        sig_past = self.sig_past[indices].clone().to(self.device)
        D_fake, G_loss = self.trainer.G_trainstep(sig_past, self.sig_future[indices].to(self.device),sig_fake_mc)
        self.training_loss['D_real'].append(D_real)
        self.training_loss['D_fake'].append(D_fake)
        self.training_loss['G_loss'].append(G_loss)
        self.evaluate(x_fake)
            
            
            
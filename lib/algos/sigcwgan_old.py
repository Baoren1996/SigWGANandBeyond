from dataclasses import dataclass

import math
import torch
from torch import autograd
from torch import optim
import torch.nn as nn
from torch.nn.utils import weight_norm

from tqdm import tqdm
from sklearn.linear_model import LinearRegression


from lib.arfnn import ResFNN
from lib.algos.base import BaseAlgo, BaseConfig
from lib.algos.gans import CGANTrainer
from lib.augmentations import SignatureConfig
from lib.augmentations import augment_path_and_compute_signatures
from lib.utils import sample_indices, to_numpy



def sigcwgan_loss(sig_pred: torch.Tensor, sig_fake_conditional_expectation: torch.Tensor):
     #torch.norm(sig_pred - sig_fake_conditional_expectation, p=10, dim=1).mean()
    return torch.norm(sig_pred - sig_fake_conditional_expectation, p=10, dim=1).mean()
    
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
    return sigs_pred


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

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=1e-2)
        self.G_scheduler = optim.lr_scheduler.StepLR(self.G_optimizer, step_size=100, gamma=0.9)

    def sample_batch(self, ):
        random_indices = sample_indices(self.sigs_pred.shape[0], self.batch_size)  # sample indices
        # sample the least squares signature and the log-rtn condition
        sigs_pred = self.sigs_pred[random_indices].clone().to(self.device)
        x_past = self.x_past[random_indices].clone().to(self.device)
        return sigs_pred, x_past

    def step(self):
        self.G.train()
        self.G_optimizer.zero_grad()  # empty 'cache' of gradients
        sigs_pred, x_past = self.sample_batch()
        sigs_fake_ce, x_fake = sample_sig_fake(self.G, self.q, self.sig_config, x_past)
        loss = sigcwgan_loss(sigs_pred, sigs_fake_ce)
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(self.G.parameters(), 10)
        self.training_loss['loss'].append(loss.item())
        self.training_loss['total_norm'].append(total_norm)
        self.G_optimizer.step()
        self.G_scheduler.step()  # decaying learning rate slowly.
        self.evaluate(x_fake)
        
class SigCWGAN_G( SigCWGAN):
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


#Modified Conditional WGAN
class CGANTrainer_MCWGAN(CGANTrainer):
    def __init__(
            self,
            G,
            D,
            G_optimizer,
            D_optimizer,
            p,
            q,
            mc_size,
            reg_param: float = 10.,
            gan_algo:str='MCWGAN_2',
            grad_reg: bool=False, 
    ):
        super(CGANTrainer_MCWGAN, self).__init__(G,D,G_optimizer,D_optimizer,p,q,gan_algo,reg_param)
        self.mc_size=mc_size
        self.grad_reg=grad_reg
        
    def G_trainstep(self, x_fake_mc, x_real):
        toggle_grad(self.G, True)
        self.G.train()
        self.G_optimizer.zero_grad()
        
        #on fake data
        d_fake_mc = self.D(x_fake_mc)
        d_fake=d_fake_mc.reshape(self.mc_size,x_real.size(0), -1).mean(0)
        
        #on real data
        d_real = self.D(x_real)
        self.D.train()
        gloss = self.compute_loss(d_real, d_fake)#gloss = self.compute_loss(d_real, d_fake)

        gloss.backward()
        self.G_optimizer.step()
        return gloss.item()

    def D_trainstep(self, x_fake_mc, x_real):
        toggle_grad(self.D, True)
        self.D.train()
        self.D_optimizer.zero_grad()

        # On real data
        x_real.requires_grad_()
        d_real = self.D(x_real)#
        #dloss_real = self.compute_loss(d_real, 1)

        # On fake data
        x_fake_mc.requires_grad_()
        d_fake_mc = self.D(x_fake_mc)
        x_fake=x_fake_mc.reshape(self.mc_size,x_real.size(0), x_real.size(1),-1).mean(0)

        
        # Compute loss
        dloss_real=torch.pow(d_real-1,2).mean() 
        dloss_fake=torch.pow(d_fake_mc,2).reshape(self.mc_size,x_real.size(0), -1).mean(0).mean()
        dloss = dloss_real+dloss_fake
        
        #dloss.backward()
        if self.grad_reg==True:
          reg = self.reg_param * self.wgan_gp_reg(x_real, x_fake)
          total_loss=dloss+reg
          total_loss.backward()
        else:
            reg = torch.zeros(1)
            total_loss=dloss
            total_loss.backward()
        #print(reg)
        # Step discriminator params
        self.D_optimizer.step()

        # Toggle gradient to False
        toggle_grad(self.D, False)
        return dloss.item(), reg.item()
        
    def compute_loss(self, d_real,d_fake):
          return  torch.pow(d_real-d_fake,2).mean()

    def wgan_gp_reg(self, x_real, x_fake, center=1.):
        batch_size = x_real.size(0)
        eps = torch.rand(batch_size, device=x_real.device).view(batch_size, 1, 1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out= self.D(x_interp)
        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()
        return reg

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

class MCWGAN_2(BaseAlgo):
    '''
    using Tan function as the activation func 
    where MCWGAN_3 use indentity function 
    '''
    def __init__(self, base_config, config: SigCWGANConfig, x_real):
        super(MCWGAN_2, self).__init__(base_config, x_real)
        
        self.sig_config = config
        self.mc_size = config.mc_size

        self.x_past = x_real[:, :self.p]
        self.x_future = x_real[:, self.p:]
        
        self.D_steps_per_G_step = 3
        self.D = D_Res(self.dim * (self.q), 1, self.hidden_dims).to(self.device)#activation to Identity
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=2e-3, betas=(0, 0.9))  # Using TTUR
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=1e-3, betas=(0, 0.9)) #lr very small
        
        self.mc_size = config.mc_size
        self.trainer = CGANTrainer_MCWGAN(  # central object to tune the GAN
            G=self.G, D=self.D, G_optimizer=self.G_optimizer, D_optimizer=self.D_optimizer, p=self.p, q=self.q,mc_size=self.mc_size
        )

    def step(self):
        for i in range(self.D_steps_per_G_step):
            # generate x_fake
            z = torch.randn(self.batch_size, self.q, self.latent_dim).to(self.device)
            indices = sample_indices(self.x_real.shape[0], self.batch_size)
            x_past = self.x_past[indices].clone().to(self.device)
            with torch.no_grad():
                  x_fake_mc = sample_x_fake(self.G, self.q, self.mc_size, x_past)
            D_loss, reg = self.trainer.D_trainstep(x_fake_mc, self.x_future[indices].to(self.device))
            if i == 0: 
                self.training_loss['D_loss'].append(D_loss)
                self.training_loss['MCWGAN_reg'].append(reg)
        # Generator step
        indices = sample_indices(self.x_real.shape[0], self.batch_size)
        x_past = self.x_past[indices].clone().to(self.device)
        x_fake_mc = sample_x_fake(self.G, self.q, self.mc_size, x_past)
        
        G_loss = self.trainer.G_trainstep(x_fake_mc,self.x_future[indices].clone().to(self.device))
        self.training_loss['D_loss'].append(D_loss)
        self.training_loss['G_loss'].append(G_loss)
        self.evaluate(x_fake_mc)
        
class MCWGAN_3(MCWGAN_2):
  def __init__(self, base_config, config: SigCWGANConfig, x_real):
        super(MCWGAN_3, self).__init__(base_config, config, x_real)

        self.D = D_Res(self.dim * (self.q), 1, self.hidden_dims,activation=nn.Identity()).to(self.device)
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=2e-3, betas=(0, 0.9))  # Using TTUR
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=1e-3, betas=(0, 0.9)) #lr very small
        ##
        #self.G_scheduler = optim.lr_scheduler.StepLR(self.G_optimizer, step_size=100, gamma=0.9)
        
class MCSWGAN(MCWGAN_2):
    #use signature as the feature set in the discriminator 
    def __init__(self, base_config, config: SigCWGANConfig, x_real):
        super(MCSWGAN, self).__init__(base_config, config,x_real)

        self.sig_config = config
        self.mc_size = config.mc_size
        config.sig_config_past.depth
        self.x_past = x_real[:, :self.p]
        self.x_future = x_real[:, self.p:]
        self.sig_future=self.sig_config.compute_sig_future(self.x_future)#compute the signature
        self.sig_dim=self.sig_future.shape[-1]
        
        self.D_steps_per_G_step = 3
        self.D = D_Res(self.sig_dim, 1, self.hidden_dims,activation=nn.Identity()).to(self.device)
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=2e-3, betas=(0, 0.9))  # Using TTUR
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=1e-3, betas=(0, 0.9)) #lr very small
        
        self.mc_size = config.mc_size
        self.trainer = CGANTrainer_MCWGAN(  # central object to tune the GAN
            G=self.G, D=self.D, G_optimizer=self.G_optimizer, D_optimizer=self.D_optimizer, p=self.p, q=self.q,mc_size=self.mc_size
        )
        
    def step(self):
        for i in range(self.D_steps_per_G_step):
            # generate x_fake
            z = torch.randn(self.batch_size, self.q, self.latent_dim).to(self.device)
            indices = sample_indices(self.x_real.shape[0], self.batch_size)
            x_past = self.x_past[indices].clone().to(self.device)
            with torch.no_grad():
                  x_fake_mc = sample_x_fake(self.G, self.q, self.mc_size, x_past)
                  sig_fake_mc=self.sig_config.compute_sig_future(x_fake_mc)
            D_loss, reg = self.trainer.D_trainstep(sig_fake_mc, self.sig_future[indices].to(self.device))
            if i == 0: 
                self.training_loss['D_loss'].append(D_loss)
                self.training_loss['MCSWGAN_reg'].append(reg)
        # Generator step]
        indices = sample_indices(self.x_real.shape[0], self.batch_size)
        x_past = self.x_past[indices].clone().to(self.device)
        x_fake_mc = sample_x_fake(self.G, self.q, self.mc_size, x_past)
        sig_fake_mc=self.sig_config.compute_sig_future(x_fake_mc)
        
        G_loss = self.trainer.G_trainstep(sig_fake_mc,self.sig_future[indices].clone().to(self.device))
        self.training_loss['D_loss'].append(D_loss)
        self.training_loss['G_loss'].append(G_loss)
        self.evaluate(x_fake_mc)


class D_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim,output_dim=1,n_layers=1):
        super(D_LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim= hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=hidden_dim, num_layers=n_layers,batch_first=True)# tanh, check the activation function
        self.linear = nn.Linear(hidden_dim, output_dim)
    def forward(self,x):
        # Input x has shape (N,T,D)
        h=self.lstm(x)[0]
        x=self.linear(h[:,-1]) 
        # (N,1)
        return x 
        
class D_Res(nn.Module):
    def __init__(self, input_dim,output_dim,hidden_dims,activation=nn.Sigmoid()):
        super(D_Res, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims= hidden_dims
        self.output_dim = output_dim
        self.Res= ResFNN(input_dim, 1, hidden_dims, True)# tanh, check the activation function
        self.activation=activation
    def forward(self,x):
        # Input x has shape (N,T,D)
        y=self.Res(x)
        z=self.activation(y)
        # (N,1)
        return z 
        

class MCWGAN_1(BaseAlgo):
    def __init__(
            self,
            base_config: BaseConfig,
            config: SigCWGANConfig,
            x_real: torch.Tensor,
    ):
        super(MCWGAN_1, self).__init__(base_config, x_real)
        self.sig_config = config
        self.mc_size = config.mc_size
        #function D
        self.D = D_Res(self.dim * (self.q), 1, self.hidden_dims).to(self.device)#ResFNN(self.dim * (self.q), 1, self.hidden_dims, True).to(self.device)#D_LSTM(self.dim, 64,output_dim=1,n_layers=1).to(self.device) 
        
        self.x_past = x_real[:, :self.p]
        self.x_future = x_real[:, self.p:]
        #
        self.D_steps_per_G_step=10
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=1e-3,betas=(0.1, 0.999))#torch.optim.RMSprop optim.RMSprop(self.G.parameters(), lr=1e-2)
        self.G_scheduler = optim.lr_scheduler.StepLR(self.G_optimizer, step_size=100, gamma=0.9)
        self.reg_param=10
        
    def sample_batch(self, ):
        random_indices = sample_indices(self.x_past.shape[0], self.batch_size)  # sample indices
        # sample the least squares signature and the log-rtn condition
        x_future=self.x_future[random_indices].clone().to(self.device)
        x_past = self.x_past[random_indices].clone().to(self.device)
        return x_past,x_future   
        
    def mcwgan_loss(self,x_future: torch.Tensor, x_fake_mc: torch.Tensor,D):
        d_real=D(x_future)
        d_fake=D(x_fake_mc).reshape(self.sig_config.mc_size,x_future.size(0), -1).mean(0)
        return torch.pow(d_real-d_fake,2).mean() 
        
    def fit(self):
        if self.batch_size == -1:
            self.batch_size = self.x_real.shape[0]
        for i in tqdm(range(self.total_steps), ncols=80):  # sig_pred, x_past, x_real
            self.step(i)
            
    def step(self,i):
        if i%10==0: 
            self.D.apply(weights_init_uniform)# resample the weights of model 
        x_past,x_future = self.sample_batch()
        #x_fake=x_fake_mc.reshape(self.mc_size,x_future.shape[0],x_future.shape[1],-1).mean(0)
        #for k in range(self.D_steps_per_G_step):
          #reg=self.D_step(x_future,x_fake)
          
        #compute the norm of gradient
        loss,loss_grad=self.G_step(x_future,x_past)
        
        self.training_loss['loss'].append(loss)
        
        #loss = self.mcwgan_loss(x_future, x_fake_mc,self.D)
        #reg=self.compute_gradnorm(loss, x_fake_mc)
        self.training_loss['Reg'].append(loss_grad)
        
        with torch.no_grad():
          x_fake_future = self.G.sample(self.q, x_past)
        self.evaluate(x_fake_future)
        
    def D_step(self,x_future,x_fake):
        x_fake.requires_grad_()
        x_future.requires_grad_()
        toggle_grad(self.D, True) 
        self.D.train() 
        self.D_optimizer.zero_grad()  
        
        reg = self.reg_param * self.wgan_gp_reg(x_future, x_fake)
        reg.backward()
        self.D_optimizer.step()

        return reg.item() 
        
    def G_step(self,x_future,x_past):
        x_future.requires_grad_()
        toggle_grad(self.G, True)
        self.G.train() 
        self.G_optimizer.zero_grad() 
        x_fake_mc = sample_x_fake(self.G, self.q, self.mc_size,x_past)
        x_fake_mc.requires_grad_()
        
        self.D.train()
        loss = self.mcwgan_loss(x_future, x_fake_mc,self.D)
        #x_fake=x_fake_mc.reshape(self.mc_size,x_future.shape[0],x_future.shape[1],-1).mean(0)
        loss_grad=0#loss.grad.pow(2).mean().item()
        loss.backward()
        self.G_optimizer.step()
        return loss.item(),loss_grad
        
    def compute_gradnorm(self, loss, x_fake, center=1.):
        batch_size = x_fake.size(0)
        reg = compute_grad2(loss, x_fake).mean()
        return reg.item()

def compute_grad2(loss, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=loss, inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2) 
    assert (grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg
        


def weights_init_uniform(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model...
        with torch.no_grad(): 
          if classname.find('Linear') !=-1:
            stdv=1./math.sqrt(m.weight.size(1))
            # apply a uniform distribution to the weights and a bias=0
            m.weight.data.uniform_(-stdv, stdv)
            m.weight.data.div(m.weight.data.norm(p=2))#norm=1
            if m.bias is not None:
              m.bias.data.uniform_(-stdv, stdv)
            
            
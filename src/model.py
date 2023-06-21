import math 
import torch
from torch import nn, Tensor
from torch.distributions import Distribution, Normal, ContinuousBernoulli, Categorical, Bernoulli, Beta
from typing import *
import torch.nn.functional as F

class ReparameterizedDiagonalGaussian(Distribution):
    """
    A distribution `N(y | mu, sigma I)` compatible with the reparameterization trick given `epsilon ~ N(0, 1)`.
    """
    def __init__(self, mu: Tensor, log_sigma:Tensor):
        assert mu.shape == log_sigma.shape, f"Tensors `mu` : {mu.shape} and ` log_sigma` : {log_sigma.shape} must be of the same shape"
        self.mu = mu
        self.sigma = log_sigma.exp()
        
    def sample_epsilon(self) -> Tensor:
        """`\eps ~ N(0, I)`"""
        return torch.empty_like(self.mu).normal_()
        
    def sample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (without gradients)"""
        with torch.no_grad():
            return self.rsample()
        
    def rsample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (with the reparameterization trick) """
        return self.mu + self.sigma * self.sample_epsilon() # <- your code
        
    def log_prob(self, z:Tensor) -> Tensor:
        """return the log probability: log `p(z)`"""
        dist = Normal(self.mu, self.sigma) # <- your code
        return dist.log_prob(z)
    
class AUX(nn.Module):
    """
    Adversarial network
    `q_\delta(y|z) = Cat(y|\pi_\delta(z))
    """
    def __init__(self, z_dim: int = 2, num_classes: int = 2) -> None:
      super(AUX, self).__init__()
      self.z_dim = z_dim
      self.num_classes = num_classes
      self.classifier = nn.Sequential(
          nn.Linear(z_dim, 64),
          nn.ReLU(),
          nn.Linear(64, 64),
          nn.ReLU(),
          nn.Linear(64, num_classes)
      )
    def posterior(self, z:Tensor) -> Distribution:
      """return the distribution `q(y|z) = Cat(y|\pi_\delta(x))`"""
      qy_logits = self.classifier(z)
      return Categorical(logits=qy_logits, validate_args=False)

    def forward(self, z) -> Distribution:
      # define the posterior q(y|z)
      qy = self.posterior(z)
      return qy
    
class nl_adversary(nn.Module):
    """
    Standard adversarial network for computing adversarial error on test set
    """
    def __init__(self, z_dim: int = 2, num_classes: int = 2) -> None:
      super(nl_adversary, self).__init__()
      self.z_dim = z_dim
      self.num_classes = num_classes
      self.classifier = nn.Sequential(
          nn.Linear(z_dim, 64),
          nn.ReLU(),
          nn.Linear(64, 64),
          nn.ReLU(),
          nn.Linear(64, num_classes)
      )
    def forward(self, z):
        return self.classifier(z)

class l_adversary(nn.Module):
    """
    Linear adversarial network for computing adversarial error on test set
    """
    def __init__(self, z_dim: int = 2, num_classes: int = 2) -> None:
      super(l_adversary, self).__init__()
      self.z_dim = z_dim
      self.num_classes = num_classes
      self.classifier = nn.Sequential(
          nn.Linear(z_dim, num_classes)
      )
    def forward(self, z) -> Tensor:
      return self.classifier(z)

class DSVAE_prior_MNIST(nn.Module):
    """Disentangled Subspace VAE combined with a classifier for the MNIST dataset with
    * a Continuous Bernoulli observation model `p_\theta(x | w, z) = CB(x | g_\theta(w, z))`
    * a Gaussian prior p(w | y = 1) = N(w | 0, 0.1*I), p(w | y = 0) = N(w | 3, I) and `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_\phi(w|x) = N(w | \mu(x), \sigma(x))` and `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    * a Bernoulli posterior for the logistic regression classifier `q_\gamma(y|w) = B(y|\pi_\gamma(w))`
    """
    
    def __init__(self, 
                 x_dim: int = 784,
                 w_dim: int = 2, 
                 z_dim: int = 2, 
                 m0: float = 3, 
                 s0: float = 1, 
                 m1: float = 0, 
                 s1: float = 0.1) -> None:
        super(DSVAE_prior_MNIST, self).__init__()  

        self.x_dim = x_dim
        self.w_dim = w_dim
        self.z_dim = z_dim   

        # Inference Network
        # Encode the observation `x` into the parameters of the posterior distribution
        # `q_\phi(w|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        self.encoderW = nn.Sequential(
            nn.Linear(x_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2*w_dim) 
        )
        # `q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        self.encoderZ = nn.Sequential(
            nn.Linear(x_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2*z_dim) 
        )
        
        # Generative Model
        # Decode the latent sample `z` and `w` into the parameters of the observation model
        # `p_\theta(x | w, z) = \prod_i CB(x_i | g_\theta(w, z))`
        self.decoder = nn.Sequential(
            nn.Linear(w_dim + z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, x_dim), 
            nn.Sigmoid()            
        )
        # Linear classifier
        # Decode the latent sample `w` into the parameters of the classifier
        # `q_\gamma(y|w) = B(y|\pi_\gamma(w))`
        self.decoderY = nn.Sequential(
            nn.Linear(w_dim, 1),
            nn.Sigmoid()
        )
        
        # define the parameters of the prior, chosen as p(z) = N(0, I)
        ones = torch.ones((1, w_dim))
        self.register_buffer('prior_params_z',  torch.zeros(torch.Size([1, 2*z_dim])))
        self.register_buffer('prior_params_w', torch.cat(
            (
                torch.cat((m0*ones, math.log(s0)*ones), 1), 
                torch.cat((m1*ones, math.log(s1)*ones), 1)
            ), 
            0
          )
        )
      
    def posteriorW(self, x:Tensor) -> Distribution:
        """return the distribution `q(w|x) = N(w | \mu(x), \sigma(x))`"""
        
        # compute the parameters of the posterior
        h_xy = self.encoderW(x)
        mu, log_sigma =  h_xy.chunk(2, dim=-1)
        
        # return a distribution `q(w|x, y) = N(w | \mu(x, y), \sigma(x, y))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
        
    def posteriorZ(self, x:Tensor) -> Distribution:
        """return the distribution `q(z|x) = N(z | \mu(x), \sigma(x))`"""
        
        # compute the parameters of the posterior
        h_x = self.encoderZ(x)
        mu, log_sigma =  h_x.chunk(2, dim=-1)
        
        # return a distribution `q(z|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def priorW(self, y, batch_size=1)-> Distribution:
        """return the distribution `p(w|y)`"""
        prior_params = torch.vstack([self.prior_params_w[int(i)] for i in y])
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def priorZ(self, batch_size:int=1)-> Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params_z.expand(batch_size, *self.prior_params_z.shape[-1:])
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def decode(self, w:Tensor, z:Tensor) -> Tensor:
        """return the parameters of the distribution `p(x|w, z)`"""
        wz = torch.cat((w, z), 1)
        return self.decoder(wz)
    
    def observation_model(self, probs:Tensor) -> Distribution:
        """return the distribution `p(x|w, z)`"""
        return ContinuousBernoulli(probs=probs, validate_args=False)    
    
    def posteriorY(self, w: Tensor):
        """return the distribution `p(y|w)`"""
        probs = self.decoderY(w)
        return Bernoulli(probs=probs, validate_args=False)

    def forward(self, x, y) -> Dict[str, Any]:
        """compute the posterior q(w|x) and q(z|x) (encoder), sample w~q(w|x), z~q(z|x) and return the distribution p(x|w,z) (decoder)"""
        
        # flatten the input
        x = x.view(x.size(0), -1)

        # define the posterior q(w|x) / encode x into q(w|x)
        qw = self.posteriorW(x)
        
        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posteriorZ(x)

        # define the prior p(w|y)
        pw = self.priorW(y, batch_size=x.size(0))
        
        # define the prior p(z)
        pz = self.priorZ(batch_size=x.size(0))
        
        # sample the posterior using the reparameterization trick: w ~ q(w | x, y), z ~ q(z | x)
        w = qw.rsample()
        z = qz.rsample()
        
        # define the observation model p(x|w, z) = B(x | g(w, z))
        px_probs = self.decode(w, z)
        px = self.observation_model(px_probs)

        # define the classifier q(y|w)
        qy = self.posteriorY(w)

        return {'px': px, 'pw': pw, 'pz': pz, 'qw': qw, 'qz': qz, 'x': px_probs, 'w': w, 'z': z, 'qy': qy}
    
    def classifier(self, x):
        with torch.no_grad():
            qw = self.posteriorW(x)
            w = qw.sample()
            y = self.posteriorY(w).sample()
            return y

class DSVAE_MNIST(nn.Module):
    """Disentangled Subspace VAE combined with a classifier for the MNIST dataset with
    * a Continuous Bernoulli observation model `p_\theta(x | w, z) = CB(x | g_\theta(w, z))`
    * a Gaussian prior `p(w) = N(w | 0, I)` and `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_\phi(w|x) = N(w | \mu(x), \sigma(x))` and `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    * a Bernoulli posterior for the logistic regression classifier `q_\gamma(y|w) = B(y|\pi_\gamma(w))`
    """
    
    def __init__(self, 
                 x_dim: int = 784,
                 w_dim: int = 2, 
                 z_dim: int = 2, 
                 m0: float = 3, 
                 s0: float = 1, 
                 m1: float = 0, 
                 s1: float = 0.1) -> None:
        super(DSVAE_prior_MNIST, self).__init__()  

        self.x_dim = x_dim
        self.w_dim = w_dim
        self.z_dim = z_dim   

        # Inference Network
        # Encode the observation `x` into the parameters of the posterior distribution
        # `q_\phi(w|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        self.encoderW = nn.Sequential(
            nn.Linear(x_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2*w_dim) 
        )
        # `q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        self.encoderZ = nn.Sequential(
            nn.Linear(x_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2*z_dim) 
        )
        
        # Generative Model
        # Decode the latent sample `z` and `w` into the parameters of the observation model
        # `p_\theta(x | w, z) = \prod_i CB(x_i | g_\theta(w, z))`
        self.decoder = nn.Sequential(
            nn.Linear(w_dim + z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, x_dim), 
            nn.Sigmoid()            
        )
        # Linear classifier
        # Decode the latent sample `w` into the parameters of the classifier
        # `q_\gamma(y|w) = B(y|\pi_\gamma(w))`
        self.decoderY = nn.Sequential(
            nn.Linear(w_dim, 1),
            nn.Sigmoid()
        )
        
        # define the parameters of the prior, chosen as p(z) = N(0, I)
        ones = torch.ones((1, w_dim))
        self.register_buffer('prior_params_z',  torch.zeros(torch.Size([1, 2*z_dim])))
        self.register_buffer('prior_params_w',  torch.zeros(torch.Size([1, 2*w_dim])))
      
    def posteriorW(self, x:Tensor) -> Distribution:
        """return the distribution `q(w|x) = N(w | \mu(x), \sigma(x))`"""
        
        # compute the parameters of the posterior
        h_xy = self.encoderW(x)
        mu, log_sigma =  h_xy.chunk(2, dim=-1)
        
        # return a distribution `q(w|x, y) = N(w | \mu(x, y), \sigma(x, y))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
        
    def posteriorZ(self, x:Tensor) -> Distribution:
        """return the distribution `q(z|x) = N(z | \mu(x), \sigma(x))`"""
        
        # compute the parameters of the posterior
        h_x = self.encoderZ(x)
        mu, log_sigma =  h_x.chunk(2, dim=-1)
        
        # return a distribution `q(z|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def priorW(self, batch_size=1)-> Distribution:
        """return the distribution `p(w)`"""
        prior_params = self.prior_params_w.expand(batch_size, *self.prior_params_w.shape[-1:])
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def priorZ(self, batch_size:int=1)-> Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params_z.expand(batch_size, *self.prior_params_z.shape[-1:])
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def decode(self, w:Tensor, z:Tensor) -> Tensor:
        """return the parameters of the distribution `p(x|w, z)`"""
        wz = torch.cat((w, z), 1)
        return self.decoder(wz)
    
    def observation_model(self, probs:Tensor) -> Distribution:
        """return the distribution `p(x|w, z)`"""
        return ContinuousBernoulli(probs=probs, validate_args=False)    
    
    def posteriorY(self, w: Tensor):
        """return the distribution `p(y|w)`"""
        probs = self.decoderY(w)
        return Bernoulli(probs=probs, validate_args=False)

    def forward(self, x, y) -> Dict[str, Any]:
        """compute the posterior q(w|x) and q(z|x) (encoder), sample w~q(w|x), z~q(z|x) and return the distribution p(x|w,z) (decoder)"""
        
        # flatten the input
        x = x.view(x.size(0), -1)

        # define the posterior q(w|x) / encode x into q(w|x)
        qw = self.posteriorW(x)
        
        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posteriorZ(x)

        # define the prior p(w)
        pw = self.priorW(batch_size=x.size(0))
        
        # define the prior p(z)
        pz = self.priorZ(batch_size=x.size(0))
        
        # sample the posterior using the reparameterization trick: w ~ q(w | x, y), z ~ q(z | x)
        w = qw.rsample()
        z = qz.rsample()
        
        # define the observation model p(x|w, z) = B(x | g(w, z))
        px_probs = self.decode(w, z)
        px = self.observation_model(px_probs)

        # define the classifier q(y|w)
        qy = self.posteriorY(w)

        return {'px': px, 'pw': pw, 'pz': pz, 'qw': qw, 'qz': qz, 'x': px_probs, 'w': w, 'z': z, 'qy': qy}
    
    def classifier(self, x):
        with torch.no_grad():
            qw = self.posteriorW(x)
            w = qw.sample()
            y = self.posteriorY(w).sample()
            return y

def reduce(x:Tensor) -> Tensor:
    """for each datapoint: sum over all dimensions"""
    return x.view(x.size(0), -1).sum(dim=1)

class VI_baseline(nn.Module):
    def __init__(self, bx, bw, bz, by):
        super().__init__()
        self.bx = bx
        self.bw = bw
        self.bz = bz
        self.by = by
        
    def forward(self, model:nn.Module, x:Tensor, y:Tensor) -> Tuple[Tensor, Dict]:
        
        # forward pass through the model
        outputs = model(x, y)
        
        # unpack outputs
        px, pw, pz, qw, qz, w, z, qy = [outputs[k] for k in ["px", "pw", "pz", "qw", "qz", "w", "z", "qy"]]
        
        # evaluate log probabilities
        log_px = reduce(px.log_prob(x))
        log_pw = reduce(pw.log_prob(w))
        log_pz = reduce(pz.log_prob(z))
        log_qw = reduce(qw.log_prob(w))
        log_qz = reduce(qz.log_prob(z))
        log_qy = reduce(qy.log_prob(y.view(-1, 1))) 

        qy = torch.exp(log_qy)         
        
        # compute the variational lower bound with beta parameters:
        # `m1 = - \beta_1 E_q [ log p(x|z) ] + \beta_2 * D_KL(q(w|x,y) | p(w|y))` + \beta_3 * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(w|x,y) | p(w|y)) = log q(w|x,y) - log p(w|y)` and `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        kl_w = log_qw - log_pw
        kl_z = log_qz - log_pz
        
        m = - self.bx*log_px + self.bw*kl_w + self.bz*kl_z - self.by*log_qy
        loss = m.mean()
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'loss': m, 'log_px':log_px, 'kl_w': kl_w, 'kl_z': kl_z, 'qy': qy}
            
        return loss, diagnostics, outputs
    
class VI_adv_marg(nn.Module):
    def __init__(self, bx, bw, bz, bhw, bhz, byw, byz, bc):
        super().__init__()
        self.bx = bx
        self.bw = bw
        self.bz = bz
        self.bhw = bhw
        self.bhz = bhz
        self.byw = byw
        self.byz = byz
        self.bc = bc
        
    def forward(self, model:nn.Module, aux_y:nn.Module, aux_c:nn.Module, x:Tensor, y:Tensor, c:Tensor) -> Tuple[Tensor, Dict]:
        
        # forward pass through the model
        outputs = model(x, y)
        
        # unpack outputs
        px, pw, pz, qw, qz, w, z, qyw = [outputs[k] for k in ["px", "pw", "pz", "qw", "qz", "w", "z", "qy"]]
        
        # evaluate log probabilities
        log_px = reduce(px.log_prob(x))
        log_pw = reduce(pw.log_prob(w))
        log_pz = reduce(pz.log_prob(z))
        log_qw = reduce(qw.log_prob(w))
        log_qz = reduce(qz.log_prob(z))
        log_qyw = reduce(qyw.log_prob(y.view(-1, 1))) # Bernoulli log prob requires float input and same shape as qyw.logits ie (64, 1)
        # log_qyw = - F.binary_cross_entropy(qyw.probs, y.view(-1, 1).float())       
        
        # compute the variational lower bound with beta parameters:
        # `m1 = - \beta_1 E_q [ log p(x|z) ] + \beta_2 * D_KL(q(w|x,y) | p(w|y))` + \beta_3 * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(w|x,y) | p(w|y)) = log q(w|x,y) - log p(w|y)` and `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        kl_w = log_qw - log_pw
        kl_z = log_qz - log_pz
        
        m1 = - self.bx*log_px + self.bw*kl_w + self.bz*kl_z - self.byw*log_qyw

        #Train the encoder to NOT predict y from z 
        qyz = aux_y(z) #not detached update the encoder!
        hz = qyz.entropy() 
        m2 = - self.bhz * hz # h = - sum(p log p)
        # log_qyz = qyz.log_prob(y)
        # exp_log_qyz = torch.exp(log_qyz)
        # m2 = self.bhz*exp_log_qyz

        #Train the encoder to NOT predict c from w
        qc = aux_c(w)
        hw = qyw.entropy() 
        m3 = - self.bhw * hw 
        # log_qc = qc.log_prob(c)
        # exp_log_qc = torch.exp(log_qc)
        # m3 = self.bhw*exp_log_qc

        m = m1 + m2 + m3
        csvaeLoss = m.mean()

        #Train the aux net to predict y from z 
        qyz = aux_y(z.detach()) #detach: to ONLY update the AUX net
        log_qyz = qyz.log_prob(y)
        exp_log_qyz = torch.exp(log_qyz)
        ny = self.byz * log_qyz
        aux_y_loss = - ny.mean()

        #Train the aux net to predict c from w 
        qc = aux_c(w.detach())
        log_qc = qc.log_prob(c)
        exp_log_qc = torch.exp(log_qc)
        nc = self.bc * log_qc
        aux_c_loss = - nc.mean()
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'m1': m1, 'log_px':log_px, 'kl_w': kl_w, 'kl_z': kl_z, 'hw': hw, 'hz': hz, 'qy': exp_log_qyz, 'qc': exp_log_qc, 'm': m, 'log_qy': log_qyw}
            
        return csvaeLoss, aux_y_loss, aux_c_loss, diagnostics, outputs
    
class VI_adv_cond(nn.Module):
    def __init__(self, bx, bw, bz, bhw, bhz, byw, byz, bc):
        super().__init__()
        self.bx = bx
        self.bw = bw
        self.bz = bz
        self.bhw = bhw
        self.bhz = bhz
        self.byw = byw
        self.byz = byz
        self.bc = bc
        
    def forward(self, model:nn.Module, aux_y:nn.Module, aux_c:nn.Module, x:Tensor, y:Tensor, c:Tensor) -> Tuple[Tensor, Dict]:
        
        # forward pass through the model
        outputs = model(x, y)
        
        # unpack outputs
        px, pw, pz, qw, qz, w, z, qyw = [outputs[k] for k in ["px", "pw", "pz", "qw", "qz", "w", "z", "qy"]]
        
        # evaluate log probabilities
        log_px = reduce(px.log_prob(x))
        log_pw = reduce(pw.log_prob(w))
        log_pz = reduce(pz.log_prob(z))
        log_qw = reduce(qw.log_prob(w))
        log_qz = reduce(qz.log_prob(z))
        log_qyw = reduce(qyw.log_prob(y.view(-1, 1))) # Bernoulli log prob requires float input and same shape as qyw.logits ie (64, 1)      
        
        # compute the variational lower bound with beta parameters:
        # `m1 = - \beta_1 E_q [ log p(x|z) ] + \beta_2 * D_KL(q(w|x,y) | p(w|y))` + \beta_3 * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(w|x,y) | p(w|y)) = log q(w|x,y) - log p(w|y)` and `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        kl_w = log_qw - log_pw
        kl_z = log_qz - log_pz
        
        m1 = - self.bx*log_px + self.bw*kl_w + self.bz*kl_z - self.byw*log_qyw

        zc = torch.cat((z, c.view(-1, 1)), dim=1)
        wy = torch.cat((w, y.view(-1, 1)), dim=1)

        #Train the encoder to NOT predict y from z and c
        qyz = aux_y(zc) #not detached update the encoder!
        hz = qyz.entropy() 
        m2 = - self.bhz * hz # h = - sum(p log p)
        # log_qyz = qyz.log_prob(y)
        # exp_log_qyz = torch.exp(log_qyz)
        # m2 = self.bhz*exp_log_qyz

        #Train the encoder to NOT predict c from w and y
        qc = aux_c(wy)
        hw = qyw.entropy() 
        m3 = - self.bhw * hw 
        # log_qc = qc.log_prob(c)
        # exp_log_qc = torch.exp(log_qc)
        # m3 = self.bhw*exp_log_qc

        m = m1 + m2 + m3
        csvaeLoss = m.mean()

        #Train the aux net to predict y from z and c
        qyz = aux_y(zc.detach())  #detach: to ONLY update the AUX net
        log_qyz = qyz.log_prob(y)
        exp_log_qyz = torch.exp(log_qyz)
        ny = self.byz * exp_log_qyz
        aux_y_loss = - ny.mean()

        #Train the aux net to predict c from w and y
        qc = aux_c(wy.detach())
        log_qc = qc.log_prob(c)
        exp_log_qc = torch.exp(log_qc)
        nc = self.bc * exp_log_qc
        aux_c_loss = - nc.mean()
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'m1': m1, 'log_px':log_px, 'kl_w': kl_w, 'kl_z': kl_z, 'hw': hw, 'hz': hz, 'qy': exp_log_qyz, 'qc': exp_log_qc, 'm': m, 'log_qy': log_qyw}
            
        return csvaeLoss, aux_y_loss, aux_c_loss, diagnostics, outputs

class VI_adv_cond_info(nn.Module):
    def __init__(self, bx, bw, bz, bhw, bhz, byw, byz, bc):
        super().__init__()
        self.bx = bx
        self.bw = bw
        self.bz = bz
        self.bhw = bhw
        self.bhz = bhz
        self.byw = byw
        self.byz = byz
        self.bc = bc
        
    def forward(self, model:nn.Module, aux_y:nn.Module, aux_c:nn.Module, x:Tensor, y:Tensor, c:Tensor) -> Tuple[Tensor, Dict]:
        
        # forward pass through the model
        outputs = model(x, y)
        
        # unpack outputs
        px, pw, pz, qw, qz, w, z, qyw = [outputs[k] for k in ["px", "pw", "pz", "qw", "qz", "w", "z", "qy"]]
        
        # evaluate log probabilities
        log_px = reduce(px.log_prob(x))
        log_pw = reduce(pw.log_prob(w))
        log_pz = reduce(pz.log_prob(z))
        log_qw = reduce(qw.log_prob(w))
        log_qz = reduce(qz.log_prob(z))
        log_qyw = reduce(qyw.log_prob(y.view(-1, 1))) # Bernoulli log prob requires float input and same shape as qyw.logits ie (64, 1)      
        
        # compute the variational lower bound with beta parameters:
        # `m1 = - \beta_1 E_q [ log p(x|z) ] + \beta_2 * D_KL(q(w|x,y) | p(w|y))` + \beta_3 * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(w|x,y) | p(w|y)) = log q(w|x,y) - log p(w|y)` and `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        kl_w = log_qw - log_pw
        kl_z = log_qz - log_pz
        
        m1 = - self.bx*log_px + self.bw*kl_w + self.bz*kl_z - self.byw*log_qyw

        zc = torch.cat((z, c.view(-1, 1)), dim=1)
        wy = torch.cat((w, y.view(-1, 1)), dim=1)

        #Train the encoder to NOT predict y from z and c
        qyz = aux_y(zc) #not detached update the encoder!
        log_qyz = qyz.log_prob(y)
        exp_log_qyz = torch.exp(log_qyz)
        m2 = self.bhz*exp_log_qyz

        #Train the encoder to NOT predict c from w and y
        qc = aux_c(wy)
        log_qc = qc.log_prob(c)
        exp_log_qc = torch.exp(log_qc)
        m3 = self.bhw*exp_log_qc

        m = m1 + m2 + m3
        csvaeLoss = m.mean()

        #Train the aux net to predict y from z and c
        qyz = aux_y(zc.detach())  #detach: to ONLY update the AUX net
        log_qyz = qyz.log_prob(y)
        exp_log_qyz = torch.exp(log_qyz)
        ny = self.byz * exp_log_qyz
        aux_y_loss = - ny.mean()

        #Train the aux net to predict c from w and y
        qc = aux_c(wy.detach())
        log_qc = qc.log_prob(c)
        exp_log_qc = torch.exp(log_qc)
        nc = self.bc * exp_log_qc
        aux_c_loss = - nc.mean()
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'m1': m1, 'log_px':log_px, 'kl_w': kl_w, 'kl_z': kl_z, 'qy': exp_log_qyz, 'qc': exp_log_qc, 'm': m, 'log_qy': log_qyw}
            
        return csvaeLoss, aux_y_loss, aux_c_loss, diagnostics, outputs

class VI_adv_dual(nn.Module):
    """
    Conditional adversarial model with one adversary per value for the conditioned attribute
    """
    def __init__(self, bx, bw, bz, bhw, bhz, byw, byz, bc):
        super().__init__()
        self.bx = bx
        self.bw = bw
        self.bz = bz
        self.bhw = bhw
        self.bhz = bhz
        self.byw = byw
        self.byz = byz
        self.bc = bc

    def adversarial_loss(self, aux: nn.Module, z:Tensor, y:Tensor):
        q = aux(z)
        log_q = reduce(q.log_prob(y))
        return torch.exp(log_q)
        
    def forward(self, model:nn.Module, aux_y_0:nn.Module, aux_c_0:nn.Module, aux_y_1:nn.Module, aux_c_1:nn.Module, x:Tensor, y:Tensor, c:Tensor) -> Tuple[Tensor, Dict]:
        
        # forward pass through the model
        outputs = model(x, y)
        
        # unpack outputs
        px, pw, pz, qw, qz, w, z, qyw = [outputs[k] for k in ["px", "pw", "pz", "qw", "qz", "w", "z", "qy"]]
        
        # evaluate log probabilities
        log_px = reduce(px.log_prob(x))
        log_pw = reduce(pw.log_prob(w))
        log_pz = reduce(pz.log_prob(z))
        log_qw = reduce(qw.log_prob(w))
        log_qz = reduce(qz.log_prob(z))
        log_qyw = reduce(qyw.log_prob(y.view(-1, 1))) # Bernoulli log prob requires float input and same shape as qyw.logits ie (64, 1)      
        
        # compute the variational lower bound with beta parameters:
        # `m1 = - \beta_1 E_q [ log p(x|z) ] + \beta_2 * D_KL(q(w|x,y) | p(w|y))` + \beta_3 * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(w|x,y) | p(w|y)) = log q(w|x,y) - log p(w|y)` and `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        kl_w = log_qw - log_pw
        kl_z = log_qz - log_pz
        
        m1 = - self.bx*log_px + self.bw*kl_w + self.bz*kl_z - self.byw*log_qyw

        z0 = z[c == 0]
        z1 = z[c == 1]
        w0 = w[y == 0]
        w1 = w[y == 1]
        y0 = y[c == 0]
        y1 = y[c == 1]
        c0 = c[y == 0]
        c1 = c[y == 1]
        nc0 = sum(c==0)
        nc1 = sum(c==1)
        ny0 = sum(y==0)
        ny1 = sum(y==1)

        #Train the encoder to NOT predict y from z given c
        aux_y0_loss = self.adversarial_loss(aux_y_0, z0, y0).mean() if ny0 != 0 else 0
        aux_y1_loss = self.adversarial_loss(aux_y_1, z1, y1).mean() if ny1 != 0 else 0
        exp_log_qyz = (ny0*aux_y0_loss + ny1*aux_y1_loss)/(ny0+ny1)
        m2 = self.bhz*exp_log_qyz

        #Train the encoder to NOT predict c from w given y
        aux_c0_loss = self.adversarial_loss(aux_c_0, w0, c0).mean() if nc0 != 0 else 0
        aux_c1_loss = self.adversarial_loss(aux_c_1, w1, c1).mean() if nc1 != 0 else 0
        exp_log_qc = (nc0*aux_c0_loss + nc1*aux_c1_loss)/(nc0+nc1)
        m3 = self.bhw*exp_log_qc

        csvaeLoss = m1.mean() + m2 + m3

        #Train the aux net to predict y from z given c
        ny_0 = self.byz*self.adversarial_loss(aux_y_0, z0.detach(), y0) if ny0 != 0 else torch.tensor([0.])
        aux_y_0_loss = - ny_0.mean()
        ny_1 = self.adversarial_loss(aux_y_1, z1.detach(), y1) if ny1 != 0 else torch.tensor([0.])
        aux_y_1_loss = - ny_1.mean()

        #Train the aux net to predict c from w given y
        nc_0 = self.bc*self.adversarial_loss(aux_c_0, w0.detach(), c0) if nc0 != 0 else torch.tensor([0.])
        aux_c_0_loss = - nc_0.mean()
        nc_1 = self.adversarial_loss(aux_c_1, w1.detach(), c1) if nc1 != 0 else torch.tensor([0.])
        aux_c_1_loss = - nc_1.mean()
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'m1': m1, 'log_px':log_px, 'kl_w': kl_w, 'kl_z': kl_z, 'qy': exp_log_qyz, 'qc': exp_log_qc, 'm': csvaeLoss, 'log_qy': log_qyw}
            
        return csvaeLoss, aux_y_0_loss, aux_c_0_loss, aux_y_1_loss, aux_c_1_loss, diagnostics, outputs

class VI_DANN(nn.Module):
    """
    Based on (Conditional) Domain-Adversarial Neural Network:
    https://github.com/facebookresearch/DomainBed/blob/main/domainbed/algorithms.py
    """
    def __init__(self, bx, bw, bz, bhw, bhz, byw, byz, bc, conditional, w_dim=2, z_dim=2):
        super().__init__()
        self.bx = bx
        self.bw = bw
        self.bz = bz
        self.bhw = bhw
        self.bhz = bhz
        self.byw = byw
        self.byz = byz
        self.bc = bc
        self.class_embeddings = nn.Embedding(2, w_dim) # 2 classes
        self.color_embeddings = nn.Embedding(2, z_dim) # 2 colors
        self.conditional = conditional
        
    def forward(self, model:nn.Module, aux_y:nn.Module, aux_c:nn.Module, x:Tensor, y:Tensor, c:Tensor) -> Tuple[Tensor, Dict]:
        
        # forward pass through the model
        outputs = model(x, y)
        
        # unpack outputs
        px, pw, pz, qw, qz, w, z, qyw = [outputs[k] for k in ["px", "pw", "pz", "qw", "qz", "w", "z", "qy"]]
        
        # evaluate log probabilities
        log_px = reduce(px.log_prob(x))
        log_pw = reduce(pw.log_prob(w))
        log_pz = reduce(pz.log_prob(z))
        log_qw = reduce(qw.log_prob(w))
        log_qz = reduce(qz.log_prob(z))
        log_qyw = reduce(qyw.log_prob(y.view(-1, 1))) # Bernoulli log prob requires float input and same shape as qyw.logits ie (64, 1)      
        
        # compute the variational lower bound with beta parameters:
        # `m1 = - \beta_1 E_q [ log p(x|z) ] + \beta_2 * D_KL(q(w|x,y) | p(w|y))` + \beta_3 * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(w|x,y) | p(w|y)) = log q(w|x,y) - log p(w|y)` and `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        kl_w = log_qw - log_pw
        kl_z = log_qz - log_pz
        
        m1 = - self.bx*log_px + self.bw*kl_w + self.bz*kl_z - self.byw*log_qyw

        if self.conditional:
            z_input = z + self.color_embeddings(c.int())
            w_input = w + self.class_embeddings(y.int())
        else:
            z_input = z
            w_input = w

        #Train the encoder to NOT predict y from z and c
        qyz = aux_y(z_input) #not detached update the encoder!
        log_qyz = qyz.log_prob(y)
        exp_log_qyz = torch.exp(log_qyz)
        m2 = self.bhz*exp_log_qyz

        #Train the encoder to NOT predict c from w and y
        qc = aux_c(w_input)
        log_qc = qc.log_prob(c)
        exp_log_qc = torch.exp(log_qc)
        m3 = self.bhw*exp_log_qc

        m = m1 + m2 + m3
        csvaeLoss = m.mean()

        #Train the aux net to predict y from z and c
        qyz = aux_y(z_input.detach())  #detach: to ONLY update the AUX net
        log_qyz = qyz.log_prob(y)
        exp_log_qyz = torch.exp(log_qyz)
        ny = self.byz * exp_log_qyz
        aux_y_loss = - ny.mean()

        #Train the aux net to predict c from w and y
        qc = aux_c(w_input.detach())
        log_qc = qc.log_prob(c)
        exp_log_qc = torch.exp(log_qc)
        nc = self.bc * exp_log_qc
        aux_c_loss = - nc.mean()
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'m1': m1, 'log_px':log_px, 'kl_w': kl_w, 'kl_z': kl_z, 'qy': exp_log_qyz, 'qc': exp_log_qc, 'm': m, 'log_qy': log_qyw}
            
        return csvaeLoss, aux_y_loss, aux_c_loss, diagnostics, outputs

def gaussian_kernel(a, b, l):
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(2).mean(2)/(depth*l**2)
    return torch.exp(-numerator)

def MMD(a, b, l):
    return gaussian_kernel(a, a, l).mean() + gaussian_kernel(b, b, l).mean() - 2*gaussian_kernel(a, b, l).mean()

class VI_MMD_marg(nn.Module):
    def __init__(self, bx, bw, bz, by, bmw, bmz, lw=1, lz=1):
        super().__init__()
        self.bx = bx
        self.bw = bw
        self.bz = bz
        self.by = by
        self.bmw = bmw
        self.bmz = bmz
        self.lw = lw
        self.lz = lz

    def set_lengthscale_(self, lw, lz):
        self.lw = lw
        self.lz = lz
        
    def forward(self, model:nn.Module, x:Tensor, y:Tensor, c:Tensor) -> Tuple[Tensor, Dict]:

        # forward pass through the model
        outputs = model(x, y)
        
        # unpack outputs
        px, pw, pz, qw, qz, w, z, qy = [outputs[k] for k in ["px", "pw", "pz", "qw", "qz", "w", "z", "qy"]]

        # evaluate log probabilities
        log_px = reduce(px.log_prob(x))
        log_pw = reduce(pw.log_prob(w))
        log_pz = reduce(pz.log_prob(z))
        log_qw = reduce(qw.log_prob(w))
        log_qz = reduce(qz.log_prob(z))
        log_qy = reduce(qy.log_prob(y.view(-1, 1)))          
        
        # compute the variational lower bound with beta parameters:
        # `m1 = - \beta_1 E_q [ log p(x|z) ] + \beta_2 * D_KL(q(w|x,y) | p(w|y))` + \beta_3 * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(w|x,y) | p(w|y)) = log q(w|x,y) - log p(w|y)` and `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        kl_w = log_qw - log_pw
        kl_z = log_qz - log_pz
        
        # Train the encoder-decoder to reconstruct and classify the images 
        m = - self.bx*log_px + self.bw*kl_w + self.bz*kl_z - self.by*log_qy
        loss_csvae = m.mean()

        z0 = z[y == 0]
        z1 = z[y == 1]
        w0 = w[c == 0]
        w1 = w[c == 1]
        # Train the encoder to minimize MMD between w and c and between z and y
        mmd_w = MMD(w0, w1, self.lw)
        mmd_z = MMD(z0, z1, self.lz)

        loss = loss_csvae + self.bmz*mmd_w + self.bmw*mmd_z
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'loss_csvae': m, 'log_px':log_px, 'kl_w': kl_w, 'kl_z': kl_z, 'log_qy': log_qy, 'mmd_w': mmd_w, 'mmd_z': mmd_z}
            
        return loss, diagnostics, outputs
    
class VI_MMD_cond(nn.Module):
    def __init__(self, bx, bw, bz, by, bmw, bmz, lw0=1, lw1=1, lz0=1, lz1=1):
        super().__init__()
        self.bx = bx
        self.bw = bw
        self.bz = bz
        self.by = by
        self.bmw = bmw
        self.bmz = bmz
        self.lw0 = lw0
        self.lw1 = lw1
        self.lz0 = lz0
        self.lz1 = lz1

    def set_lengthscale_(self, lw0, lw1, lz0, lz1):
        self.lw0 = lw0
        self.lw1 = lw1
        self.lz0 = lz0
        self.lz1 = lz1
        
    def forward(self, model:nn.Module, x:Tensor, y:Tensor, c:Tensor) -> Tuple[Tensor, Dict]:

        # forward pass through the model
        outputs = model(x, y)
        
        # unpack outputs
        px, pw, pz, qw, qz, w, z, qy = [outputs[k] for k in ["px", "pw", "pz", "qw", "qz", "w", "z", "qy"]]

        # evaluate log probabilities
        log_px = reduce(px.log_prob(x))
        log_pw = reduce(pw.log_prob(w))
        log_pz = reduce(pz.log_prob(z))
        log_qw = reduce(qw.log_prob(w))
        log_qz = reduce(qz.log_prob(z))
        log_qy = reduce(qy.log_prob(y.view(-1, 1)))          
        
        # compute the variational lower bound with beta parameters:
        # `m1 = - \beta_1 E_q [ log p(x|z) ] + \beta_2 * D_KL(q(w|x,y) | p(w|y))` + \beta_3 * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(w|x,y) | p(w|y)) = log q(w|x,y) - log p(w|y)` and `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        kl_w = log_qw - log_pw
        kl_z = log_qz - log_pz
        
        # Train the encoder-decoder to reconstruct and classify the images 
        m = - self.bx*log_px + self.bw*kl_w + self.bz*kl_z - self.by*log_qy
        loss_csvae = m.mean()

        ny0 = sum(y==0)
        ny1 = sum(y==1)
        nc0 = sum(c==0)
        nc1 = sum(c==1)
        # Train the encoder to minimize MMD between w and c given y and between z and y given c
        MMD_w0 = MMD(w[(y == 0) & (c == 0)], w[(y == 0) & (c == 1)], self.lw0)
        MMD_w1 = MMD(w[(y == 1) & (c == 0)], w[(y == 1) & (c == 1)], self.lw1)
        MMD_z0 = MMD(z[(c == 0) & (y == 0)], z[(c == 0) & (y == 1)], self.lz0)
        MMD_z1 = MMD(z[(c == 1) & (y == 0)], z[(c == 1) & (y == 1)], self.lz1)
        MMD_w = (ny0*MMD_w0 + ny1*MMD_w1)/(ny0 + ny1)
        MMD_z = (nc0*MMD_z0 + nc1*MMD_z1)/(nc0 + nc1)

        loss = loss_csvae + self.bmz*MMD_w + self.bmw*MMD_z
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'loss_csvae': m, 'log_px':log_px, 'kl_w': kl_w, 'kl_z': kl_z, 'log_qy': log_qy, 'mmd_w': MMD_w, 'mmd_z': MMD_z}
            
        return loss, diagnostics, outputs
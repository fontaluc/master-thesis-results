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
    def __init__(self, z_dim: int = 2, num_classes: int = 2, linear: bool = False) -> None:
      super(AUX, self).__init__()
      self.z_dim = z_dim
      self.num_classes = num_classes
      self.classifier = nn.Sequential(
          nn.Linear(z_dim, 64),
          nn.ReLU(),
          nn.Linear(64, 64),
          nn.ReLU(),
          nn.Linear(64, num_classes)
      ) if not linear else nn.Linear(z_dim, num_classes)
    def posterior(self, z:Tensor) -> Distribution:
      """return the distribution `q(y|z) = Cat(y|\pi_\delta(x))`"""
      qy_logits = self.classifier(z)
      return Categorical(logits=qy_logits, validate_args=False)

    def forward(self, z) -> Distribution:
      # define the posterior q(y|z)
      qy = self.posterior(z)
      return qy

class AUX_W(nn.Module):
    """
    Auxiliary network
    `q_\delta(w|y) = N(\mu_\delta(y)), \sigma_\delta(y))
    """
    def __init__(self, y_dim: int = 1, w_dim: int = 2, linear: bool = False) -> None:
      super(AUX_W, self).__init__()
      self.y_dim = y_dim
      self.w_dim = w_dim
      self.layers = nn.Sequential(
          nn.Linear(y_dim, 64),
          nn.ReLU(),
          nn.Linear(64, 64),
          nn.ReLU(),
          nn.Linear(64, 2*w_dim)
      ) if not linear else nn.Linear(z_dim, num_classes)
    def posterior(self, y:Tensor) -> Distribution:
      """return the distribution `q(w|y) = N(\mu_\delta(y)), \sigma_\delta(y))`"""
      h_y = self.layers(y)
      mu, log_sigma = h_y.chunk(2, dim=-1)
      return Normal(mu, log_sigma.exp())

    def forward(self, y) -> Distribution:
      # define the posterior q(w|y)
      qw = self.posterior(y)
      return qw
    
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

class DSCVAE_prior_MNIST(nn.Module):
    """Disentangled Subspace CVAE combined with a classifier for the MNIST dataset with
    * a Continuous Bernoulli observation model `p_\theta(x | w, z) = CB(x | g_\theta(w, z))`
    * a Gaussian prior p(w | y = 1) = N(w | 0, I), p(w | y = 0) = N(w | 6, I) and `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_\phi(w|x) = N(w | \mu(x), \sigma(x))` and `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    * a Bernoulli posterior for the logistic regression classifier `q_\gamma(y|w) = B(y|\pi_\gamma(w))`
    """
    
    def __init__(self, 
                 w_dim: int = 2, 
                 z_dim: int = 2, 
                 m0: float = 6, 
                 s0: float = 1, 
                 m1: float = 0, 
                 s1: float = 1) -> None:
        super(DSCVAE_prior_MNIST, self).__init__()  

        self.w_dim = w_dim
        self.z_dim = z_dim   

        # Inference Network
        # Encode the observation `x` into the parameters of the posterior distribution
        # `q_\phi(w|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        self.encoderW = nn.Sequential(
            # input is cmnist image: 2x14x14
            nn.Conv2d(2, 32, 4, 2, 1),  # 32x7x7
            nn.BatchNorm2d(32),  # 32x7x7
            nn.ReLU(inplace=True),  # 32x7x7
            nn.Conv2d(32, 128, 7, 1, 0),  # 128x1x1
            nn.BatchNorm2d(128),  # 128x1x1
            nn.ReLU(inplace=True),  # 128x1x1
            nn.Conv2d(128, 512, 1, 1, 0),  # 512x1x1
            nn.BatchNorm2d(512),  # 512x1x1
            nn.ReLU(inplace=True),  # 512x1x1
            nn.Conv2d(512, 200, 1, 1, 0),  # 200x1x1
        )
        self.fcW = nn.Linear(200, 2*w_dim)

        # `q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        self.encoderZ = nn.Sequential(
            # input is cmnist image: 2x14x14
            nn.Conv2d(2, 32, 4, 2, 1),  # 32x7x7
            nn.BatchNorm2d(32),  # 32x7x7
            nn.ReLU(inplace=True),  # 32x7x7
            nn.Conv2d(32, 128, 7, 1, 0),  # 128x1x1
            nn.BatchNorm2d(128),  # 128x1x1
            nn.ReLU(inplace=True),  # 128x1x1
            nn.Conv2d(128, 512, 1, 1, 0),  # 512x1x1
            nn.BatchNorm2d(512),  # 512x1x1
            nn.ReLU(inplace=True),  # 512x1x1
            nn.Conv2d(512, 200, 1, 1, 0),  # 200x1x1
        )
        self.fcZ = nn.Linear(200, 2*z_dim)
        
        # Generative Model
        # Decode the latent sample `z` and `w` into the parameters of the observation model
        # `p_\theta(x | w, z) = \prod_i CB(x_i | g_\theta(w, z))`
        self.decoder = nn.Sequential(
            # input: (w_dim + z_dim) x 1 x 1
            nn.ConvTranspose2d(w_dim + z_dim, 512, 1, 1, 0),  # 512x1x1
            nn.BatchNorm2d(512),  # 512x1x1
            nn.ReLU(inplace=True),  # 512x1x1
            nn.ConvTranspose2d(512, 128, 1, 1, 0),  # 128x1x1
            nn.BatchNorm2d(128),  # 128x1x1
            nn.ReLU(inplace=True),  # 128x1x1
            nn.ConvTranspose2d(128, 32, 7, 1, 0),  # 32x7x7
            nn.BatchNorm2d(32),  # 32x7x7
            nn.ReLU(inplace=True),  # 32x7x7
            nn.ConvTranspose2d(32, 2, 4, 2, 1)  # 2x14x14          
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
        h_xy = self.encoderW(x.view(-1, 2, 14, 14)).squeeze()
        mu, log_sigma =  self.fcW(F.relu(h_xy)).chunk(2, dim=-1)
        
        # return a distribution `q(w|x, y) = N(w | \mu(x, y), \sigma(x, y))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
        
    def posteriorZ(self, x:Tensor) -> Distribution:
        """return the distribution `q(z|x) = N(z | \mu(x), \sigma(x))`"""
        
        # compute the parameters of the posterior
        h_x = self.encoderZ(x.view(-1, 2, 14, 14)).squeeze()
        mu, log_sigma =  self.fcZ(F.relu(h_x)).chunk(2, dim=-1)
        
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
        h =  self.decoder(wz.view(wz.size(0), wz.size(1), 1, 1))
        return torch.sigmoid(h.view(-1, 2 * 14 * 14))
    
    def observation_model(self, probs:Tensor) -> Distribution:
        """return the distribution `p(x|w, z)`"""
        return ContinuousBernoulli(probs=probs, validate_args=False)    
    
    def posteriorY(self, w: Tensor):
        """return the distribution `p(y|w)`"""
        probs = self.decoderY(w)
        return Bernoulli(probs=probs, validate_args=False)

    def forward(self, x, y) -> Dict[str, Any]:
        """compute the posterior q(w|x) and q(z|x) (encoder), sample w~q(w|x), z~q(z|x) and return the distribution p(x|w,z) (decoder)"""

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
        super(DSVAE_MNIST, self).__init__()  

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
        hw = qc.entropy() 
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
        hw = qc.entropy() 
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
        ny = self.byz * log_qyz
        aux_y_loss = - ny.mean()

        #Train the aux net to predict c from w and y
        qc = aux_c(wy.detach())
        log_qc = qc.log_prob(c)
        exp_log_qc = torch.exp(log_qc)
        nc = self.bc * log_qc
        aux_c_loss = - nc.mean()
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'m1': m1, 'log_px':log_px, 'kl_w': kl_w, 'kl_z': kl_z, 'hw': hw, 'hz': hz, 'qy': exp_log_qyz, 'qc': exp_log_qc, 'm': m, 'log_qy': log_qyw}
            
        return csvaeLoss, aux_y_loss, aux_c_loss, diagnostics, outputs


class VI_sufficiency(nn.Module):
    def __init__(self, bx, bw, bz, bhw, bhz, by):
        super().__init__()
        self.bx = bx
        self.bw = bw
        self.bz = bz
        self.bhw = bhw
        self.bhz = bhz
        self.by = by
        
    def forward(self, model:nn.Module, aux_w:nn.Module, aux_wzc:nn.Module, aux_z:nn.Module, aux_wzy:nn.Module, x:Tensor, y:Tensor, c:Tensor) -> Tuple[Tensor, Dict]:
        
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
        log_qy = reduce(qy.log_prob(y.view(-1, 1))) # Bernoulli log prob requires float input and same shape as qyw.logits ie (64, 1)      
        
        # compute the variational lower bound with beta parameters:
        # `m1 = - \beta_1 E_q [ log p(x|z) ] + \beta_2 * D_KL(q(w|x,y) | p(w|y))` + \beta_3 * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(w|x,y) | p(w|y)) = log q(w|x,y) - log p(w|y)` and `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        kl_w = log_qw - log_pw
        kl_z = log_qz - log_pz
        
        m1 = - self.bx*log_px + self.bw*kl_w + self.bz*kl_z - self.by*log_qy

        wzc = torch.cat((w, z, c.view(-1, 1)), dim=1)
        wzy = torch.cat((w, z, y.view(-1, 1)), dim=1)

        #Train the encoder for predicting y
        qw = aux_w(w) #not detached update the encoder!
        qwzc = aux_wzc(wzc)
        hw = qw.entropy()
        hwzc = qwzc.entropy() 
        dhw = hw - hwzc
        m3 = self.bhw * dhw # h = - sum(p log p)

        #Train the encoder for predicting c
        qz = aux_z(z) #not detached update the encoder!
        qwzy = aux_wzy(wzy)
        hz = qz.entropy()
        hwzy = qwzy.entropy() 
        dhz = hz - hwzy
        m2 = self.bhz * dhz # h = - sum(p log p)

        m = m1 + m2 + m3
        csvaeLoss = m.mean()

        #Train the aux net to predict y from w
        qw = aux_w(w.detach())
        log_qw = qw.log_prob(y)
        exp_log_qw = torch.exp(log_qw)
        aux_w_loss = - log_qw.mean()

        #Train the aux net to predict y from w, z, c
        qwzc = aux_wzc(wzc.detach())
        log_qwzc = qwzc.log_prob(y)
        exp_log_qwzc = torch.exp(log_qwzc)
        aux_wzc_loss = - log_qwzc.mean()

        #Train the aux net to predict c from z
        qz = aux_z(z.detach())
        log_qz = qz.log_prob(c)
        exp_log_qz = torch.exp(log_qz)
        aux_z_loss = - log_qz.mean()

        #Train the aux net to predict c from w, z, y
        qwzy = aux_wzy(wzy.detach())
        log_qwzy = qwzy.log_prob(c)
        exp_log_qwzy = torch.exp(log_qwzy)
        aux_wzy_loss = - log_qwzy.mean()
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'m1': m1, 'log_px':log_px, 'kl_w': kl_w, 'kl_z': kl_z, 'dhw': dhw, 'dhz': dhz, 'hw': hw, 'hz': hz, 'hwzc': hwzc, 'hwzy': hwzy, 'qw': exp_log_qw, 'qwzy': exp_log_qwzy, 'qz': exp_log_qz, 'qwzc': exp_log_qwzc, 'm': m, 'log_qy': log_qy}
            
        return csvaeLoss, aux_w_loss, aux_wzc_loss, aux_z_loss, aux_wzy_loss, diagnostics, outputs

class VI_sufficiency_v2(nn.Module):
    def __init__(self, bx, bw, bz, bhw, bhz, by):
        super().__init__()
        self.bx = bx
        self.bw = bw
        self.bz = bz
        self.bhw = bhw
        self.bhz = bhz
        self.by = by
        
    def forward(
        self, 
        model:nn.Module, 
        aux_w:nn.Module, 
        aux_wz:nn.Module, 
        aux_wc:nn.Module, 
        aux_z:nn.Module, 
        aux_zw:nn.Module, 
        aux_zy:nn.Module, 
        x:Tensor, 
        y:Tensor, 
        c:Tensor
        ) -> Tuple[Tensor, Dict]:
        
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
        log_qy = reduce(qy.log_prob(y.view(-1, 1))) # Bernoulli log prob requires float input and same shape as qyw.logits ie (64, 1)      
        
        # compute the variational lower bound with beta parameters:
        # `m1 = - \beta_1 E_q [ log p(x|z) ] + \beta_2 * D_KL(q(w|x,y) | p(w|y))` + \beta_3 * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(w|x,y) | p(w|y)) = log q(w|x,y) - log p(w|y)` and `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        kl_w = log_qw - log_pw
        kl_z = log_qz - log_pz
        
        m1 = - self.bx*log_px + self.bw*kl_w + self.bz*kl_z - self.by*log_qy

        wz = torch.cat((w, z), dim=1)
        wc = torch.cat((w, c.view(-1, 1)), dim=1)
        zy = torch.cat((z, y.view(-1, 1)), dim=1)

        #Train the encoder for predicting y
        qw = aux_w(w) #not detached update the encoder!
        qwz = aux_wz(wz)
        hw = qw.entropy()
        hwz = qwz.entropy() 
        dhwz = hw - hwz
        qwc = aux_wc(wc)
        hwc = qwc.entropy()
        dhwc = hw - hwc
        dhw = dhwz + dhwc
        m3 = self.bhw * dhw 

        #Train the encoder for predicting c
        qz = aux_z(z) #not detached update the encoder!
        qzw = aux_zw(wz)
        hz = qz.entropy()
        hzw = qzw.entropy() 
        dhzw = hz - hzw
        qzy = aux_zy(zy)
        hzy = qzy.entropy()
        dhzy = hz - hzy
        dhz = dhzw + dhzy
        m2 = self.bhz * dhz 

        m = m1 + m2 + m3
        csvaeLoss = m.mean()

        #Train the aux net to predict y from w
        qw = aux_w(w.detach())
        log_qw = qw.log_prob(y)
        exp_log_qw = torch.exp(log_qw)
        aux_w_loss = - log_qw.mean()

        #Train the aux net to predict y from w, z
        qwz = aux_wz(wz.detach())
        log_qwz = qwz.log_prob(y)
        exp_log_qwz = torch.exp(log_qwz)
        aux_wz_loss = - log_qwz.mean()

        #Train the aux net to predict y from w, c
        qwc = aux_wc(wc.detach())
        log_qwc = qwc.log_prob(y)
        exp_log_qwc = torch.exp(log_qwc)
        aux_wc_loss = - log_qwc.mean()

        #Train the aux net to predict c from z
        qz = aux_z(z.detach())
        log_qz = qz.log_prob(c)
        exp_log_qz = torch.exp(log_qz)
        aux_z_loss = - log_qz.mean()

        #Train the aux net to predict c from w, z
        qzw = aux_zw(wz.detach())
        log_qzw = qzw.log_prob(c)
        exp_log_qzw = torch.exp(log_qzw)
        aux_zw_loss = - log_qzw.mean()

        #Train the aux net to predict c from z, y
        qzy = aux_zy(zy.detach())
        log_qzy = qzy.log_prob(c)
        exp_log_qzy = torch.exp(log_qzy)
        aux_zy_loss = - log_qzy.mean()
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {
                'm1': m1, 
                'log_px':log_px, 
                'kl_w': kl_w, 
                'kl_z': kl_z, 
                'dhw': dhw, 
                'dhz': dhz, 
                'dhwz': dhwz,
                'dhwc': dhwc,
                'dhzw': dhzw,
                'dhzy': dhzy,
                'hw': hw, 
                'hz': hz, 
                'hwz': hwz,
                'hzw': hzw, 
                'hwc': hwc,
                'hzy': hzy,  
                'm': m, 
                'log_qy': log_qy}
            
        return csvaeLoss, aux_w_loss, aux_wz_loss, aux_wc_loss, aux_z_loss, aux_zw_loss, aux_zy_loss, diagnostics, outputs

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
        m2 = self.bhz*log_qyz

        #Train the encoder to NOT predict c from w and y
        qc = aux_c(wy)
        log_qc = qc.log_prob(c)
        m3 = self.bhw*log_qc

        m = m1 + m2 + m3
        csvaeLoss = m.mean()

        #Train the aux net to predict y from z and c
        qyz = aux_y(zc.detach())  #detach: to ONLY update the AUX net
        log_qyz = qyz.log_prob(y)
        exp_log_qyz = torch.exp(log_qyz)
        ny = self.byz * log_qyz
        aux_y_loss = - ny.mean()

        #Train the aux net to predict c from w and y
        qc = aux_c(wy.detach())
        log_qc = qc.log_prob(c)
        exp_log_qc = torch.exp(log_qc)
        nc = self.bc * log_qc
        aux_c_loss = - nc.mean()
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'m1': m1, 'log_px':log_px, 'kl_w': kl_w, 'kl_z': kl_z, 'qy': exp_log_qyz, 'qc': exp_log_qc, 'm': m, 'log_qy': log_qyw}
            
        return csvaeLoss, aux_y_loss, aux_c_loss, diagnostics, outputs

class VI_adv_dual(nn.Module):
    """
    Conditional adversarial model with one adversary per value for the conditioned attribute using the standard adversarial training
    approach 
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
        return log_q
        
    def forward(self, model:nn.Module, aux_y_0:nn.Module, aux_c_0:nn.Module, aux_y_1:nn.Module, aux_c_1:nn.Module, x:Tensor, y:Tensor, c:Tensor, device:torch.device) -> Tuple[Tensor, Dict]:
        
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
        ny0 = len(y0)
        ny1 = len(y1)
        nc0 = len(c0)
        nc1 = len(c1)

        # When there is no sample with the corresponding attribute value, set the corresponding loss to 0
        # requires_grad is set to True to enable backward prop when training the auxiliary network
        zero_tensor = torch.tensor([0.], device=device, requires_grad=True)

        #Train the encoder to NOT predict y from z given c
        log_qyz_y0 = self.adversarial_loss(aux_y_0, z0, y0).mean() if ny0 != 0 else zero_tensor
        log_qyz_y1 = self.adversarial_loss(aux_y_1, z1, y1).mean() if ny1 != 0 else zero_tensor
        log_qyz = (ny0*log_qyz_y0 + ny1*log_qyz_y1)/(ny0+ny1)
        qyz = torch.exp(log_qyz)
        m2 = self.bhz*log_qyz

        #Train the encoder to NOT predict c from w given y
        log_qc_c0 = self.adversarial_loss(aux_c_0, w0, c0).mean() if nc0 != 0 else zero_tensor
        log_qc_c1 = self.adversarial_loss(aux_c_1, w1, c1).mean() if nc1 != 0 else zero_tensor
        log_qc = (nc0*aux_c0_loss + nc1*aux_c1_loss)/(nc0+nc1)
        qc = torch.exp(log_qc)
        m3 = self.bhw*exp_log_qc

        csvaeLoss = m1.mean() + m2 + m3

        #Train the aux net to predict y from z given c
        aux_y_0_loss = - self.byz*self.adversarial_loss(aux_y_0, z0.detach(), y0).mean() if ny0 != 0 else zero_tensor
        aux_y_1_loss = - self.byz*self.adversarial_loss(aux_y_1, z1.detach(), y1).mean() if ny1 != 0 else zero_tensor

        #Train the aux net to predict c from w given y
        aux_c_0_loss = - self.bc*self.adversarial_loss(aux_c_0, w0.detach(), c0).mean() if nc0 != 0 else zero_tensor
        aux_c_1_loss= - self.bc*self.adversarial_loss(aux_c_1, w1.detach(), c1).mean() if nc1 != 0 else zero_tensor
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'m1': m1, 'log_px':log_px, 'kl_w': kl_w, 'kl_z': kl_z, 'qy': qyz, 'qc': qc, 'm': csvaeLoss, 'log_qy': log_qyw}
            
        return csvaeLoss, aux_y_0_loss, aux_c_0_loss, aux_y_1_loss, aux_c_1_loss, diagnostics, outputs

class VI_DANN(nn.Module):
    """
    Conditional adversarial model based on (Conditional) Domain-Adversarial Neural Network:
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
        m2 = self.bhz*log_qyz

        #Train the encoder to NOT predict c from w and y
        qc = aux_c(w_input)
        log_qc = qc.log_prob(c)
        m3 = self.bhw*log_qc

        m = m1 + m2 + m3
        csvaeLoss = m.mean()

        #Train the aux net to predict y from z and c
        qyz = aux_y(z_input.detach())  #detach: to ONLY update the AUX net
        log_qyz = qyz.log_prob(y)
        exp_log_qyz = torch.exp(log_qyz)
        ny = self.byz * log_qyz
        aux_y_loss = - ny.mean()

        #Train the aux net to predict c from w and y
        qc = aux_c(w_input.detach())
        log_qc = qc.log_prob(c)
        exp_log_qc = torch.exp(log_qc)
        nc = self.bc * log_qc
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
            diagnostics = {'loss_csvae': m, 'log_px':log_px, 'kl_w': kl_w, 'kl_z': kl_z, 'qy': torch.exp(log_qy), 'mmd_w': mmd_w, 'mmd_z': mmd_z}
            
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
            diagnostics = {'loss_csvae': m, 'log_px':log_px, 'kl_w': kl_w, 'kl_z': kl_z, 'qy': torch.exp(log_qy), 'mmd_w': MMD_w, 'mmd_z': MMD_z}
            
        return loss, diagnostics, outputs

class VI_MMD_cond_aux(nn.Module):
    def __init__(self, bx, bw, bz, by, bmw, bmz, bh, lw0=1, lw1=1, lz0=1, lz1=1):
        super().__init__()
        self.bx = bx
        self.bw = bw
        self.bz = bz
        self.by = by
        self.bmw = bmw
        self.bmz = bmz
        self.bh = bh
        self.lw0 = lw0
        self.lw1 = lw1
        self.lz0 = lz0
        self.lz1 = lz1

    def set_lengthscale_(self, lw0, lw1, lz0, lz1):
        self.lw0 = lw0
        self.lw1 = lw1
        self.lz0 = lz0
        self.lz1 = lz1
        
    def forward(self, model:nn.Module, aux_y:nn.Module, aux_zy:nn.Module, x:Tensor, y:Tensor, c:Tensor) -> Tuple[Tensor, Dict]:

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
        m1 = - self.bx*log_px + self.bw*kl_w + self.bz*kl_z - self.by*log_qy

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

        zy = torch.cat((z, y.view(-1, 1)), dim=1)
        #Train the encoder for predicting w
        qy = aux_y(y.view(-1, 1)) #not detached update the encoder!
        qzy = aux_zy(zy)
        hy = reduce(qy.entropy())
        hzy = reduce(qzy.entropy())
        dh = hy - hzy
        m2 = self.bh*dh

        m = m1 + m2
        loss_csvae_aux = m.mean()

        loss = loss_csvae_aux + self.bmw*MMD_w + self.bmz*MMD_z

        #Train the aux net to predict w from y
        qy = aux_y(y.view(-1, 1))
        log_qy = reduce(qy.log_prob(w.detach()))
        exp_log_qy = torch.exp(log_qy)
        aux_y_loss = - log_qy.mean()

        qzy = aux_zy(zy.detach())
        log_qzy = reduce(qzy.log_prob(w.detach()))
        exp_log_qzy = torch.exp(log_qzy)
        aux_zy_loss = - log_qzy.mean()
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'m1': m1,'log_px':log_px, 'kl_w': kl_w, 'kl_z': kl_z, 'qy': torch.exp(log_qy), 'mmd_w': MMD_w, 'mmd_z': MMD_z, 'hy': hy, 'hzy': hzy, 'dh': dh, 'qw_y': exp_log_qy, 'qw_zy': exp_log_qzy}
            
        return loss, aux_y_loss, aux_zy_loss, diagnostics, outputs
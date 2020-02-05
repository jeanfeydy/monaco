import numpy as np
import torch
from matplotlib import pyplot as plt
from pykeops.torch  import LazyTensor

import torch.nn.functional as F

from scipy.special import gamma
from .proposals import Proposal


numpy = lambda x : x.cpu().numpy()

def squared_distances(x, y):
    x_i = LazyTensor( x[:,None,:] )  # (N,1,D)
    y_j = LazyTensor( y[None,:,:] )  # (1,M,D)
    return ((x_i - y_j)**2).sum(dim=2)  # (N,M,1)


def normalize(points):
    return F.normalize(points, p=2, dim=1)



class EuclideanSpace(object):

    def __init__(self, dimension = 1, dtype = None, resolution = 200):

        self.dimension = dimension
        
        if dtype is None: dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.dtype = dtype

        # Display: create a uniform grid on the unit square
        self.resolution = resolution
        ticks = np.linspace(0, 1, self.resolution + 1)[:-1] + .5 / self.resolution

        if self.dimension == 1:
            self.grid = torch.from_numpy( ticks ).type(dtype).view(-1,1)
        else:
            x_g, y_g = np.meshgrid(ticks, ticks)
            self.grid = torch.from_numpy( np.vstack((x_g.ravel(), y_g.ravel())).T ).type(dtype).contiguous()

    
    def apply_noise(self, x, v):
        return x + v


    def discrepancy(self, x, y):
        """Computes the energy distance between two samples."""

        n, m = len(x), len(y) 

        D_xx = squared_distances(x, x).sqrt().sum(dim=1).sum()
        D_xy = squared_distances(x, y).sqrt().sum(dim=1).sum()
        D_yy = squared_distances(y, y).sqrt().sum(dim=1).sum()

        return D_xy / (n*m) - .5 * (D_xx / (n*n) + D_yy / (m*m))
        
    
    def scatter(self, points, color, ax = None):

        if ax is None: ax = plt.gca()

        if self.dimension == 1:  # display as a log-histogram
            ax.hist(numpy(points), bins = self.resolution, range = (0,1), 
                    color = color, histtype="step", density=True, log=True)

        else:  # display as a good old point cloud
            xy = numpy(points[:,:2])
            ax.scatter(xy[:, 0], xy[:, 1], 200 / len(xy), color=color, zorder=4)


    def plot(self, potential, color, ax = None):

        if ax is None: ax = plt.gca()

        if self.dimension == 1:
            log_heatmap = numpy(potential(self.grid))
            ax.plot(numpy(self.grid).ravel(), np.exp(-log_heatmap.ravel()), color = color)

        elif self.dimension == 2:
            log_heatmap = numpy(potential(self.grid))
            log_heatmap = log_heatmap.reshape(self.resolution, self.resolution)

            scale  = np.amax(np.abs(log_heatmap[:]))
            levels = np.linspace(-scale, scale, 41)

            ax.contour(log_heatmap, origin='lower', linewidths=1., colors=color,
                        levels=levels, extent=(0, 1, 0, 1))
        else:
            None

    
    def draw_frame(self, ax = None):
        if ax is None: ax = plt.gca()

        if self.dimension == 1:
            ax.axis([0,1,1e-5,1e2])
        else:  
            ax.plot([0,1,1,0,0], [0,0,1,1,0], 'k')  # unit square
            ax.axis("equal")
            ax.axis("off")




class Mixture(object):
    def __init__(self, space, means, deviations, weights):

        self.D = space.dimension
        self.dtype = space.dtype
        self.m = means
        self.s = deviations
        self.w = weights


    def potential(self, x):
        """Evaluates the potential on the point cloud x."""

        D_ij    = squared_distances(x, self.m)  # (N,M,1)
        s_j = LazyTensor( self.s[None,:,None])  # (1,M,1)
        w_j = LazyTensor( self.w[None,:,None])  # (1,M,1)

        V_i = self.log_density(D_ij, s_j, w_j)
        return V_i.reshape(-1)  # (N,)


    def sample(self, N = 1):
        """Returns a sample array of shape (N,D)."""
        y = self.sample_noise(N)
        classes = np.random.choice(len(self.w), N, p = numpy(self.w))
        y = self.m[classes] + self.s[classes].view(-1,1) * y
        return y


    def log_density(self, D_ij, s_j, w_j):
        raise NotImplementedError()


    def sample_noise(self, N = 1):
        raise NotImplementedError()



class GaussianMixture(Mixture):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    def log_density(self, D_ij, s_j, w_j):
        D = self.D

        logK_ij = - D_ij / (2 * s_j**2) + w_j.log() \
                  - (D/2) * float( np.log(2 * np.pi) ) - D * s_j.log()  # (N,M,1)
        V_i = - logK_ij.logsumexp(dim=1)  # (N,1), genuine torch Tensor

        return V_i.reshape(-1)  # (N,)


    def sample_noise(self, N = 1):
        return torch.randn(N, self.D).type(self.dtype)



class CauchyMixture(Mixture):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.D != 1:
            raise NotImplementedError()
    

    def log_density(self, D_ij, s_j, w_j):
        D = self.D

        K_ij = 1 / ( float(np.pi) * s_j * (1 + D_ij / (s_j**2) ) )  # (N,M,1)
        V_i = - K_ij.sum(dim=1).log()  # (N,1), genuine torch Tensor

        return V_i.reshape(-1)  # (N,)


    def sample_noise(self, N = 1):
        y = torch.rand(N, self.D).type(self.dtype)
        return (np.pi * (y - .5)).tan()





class UnitPotential(object):
    def __init__(self, space, potential):

        self.D = space.dimension
        self.dtype = space.dtype
        self.inner_potential = potential


    def potential(self, x):
        """Evaluates the potential on the point cloud x."""

        V_i = self.inner_potential(x)

        out_of_bounds = (x - .5).abs().max(dim=1).values > .5
        V_i[out_of_bounds] = 10000000.

        return V_i.reshape(-1)  # (N,)


    def sample(self, N = 1):
        """Returns a sample array of shape (N,D)."""
        x = torch.rand(N, self.D).type(self.dtype)
        uniform = torch.rand(N).type(self.dtype)
        
        threshold = (- self.potential(x)).exp()
        reject = (uniform > threshold).view(-1)
        M = int(reject.sum())

        if M == 0:
            return x
        else:
            x[reject] = self.sample(M)
            return x







class BallProposal(Proposal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def sample_noise(self, N, scales):

        r = torch.rand(N, 1).type(self.dtype) ** (1 / self.D)  # Radius
        n = torch.randn(N, self.D).type(self.dtype)
        n = normalize(n)  # Direction, randomly sampled on the sphere

        return scales * r * n


    def nlog_density(self, target, source, log_weights, scales, probas):

        D_ij = squared_distances(target, source)
        neighbors_ij = (scales ** 2 - D_ij).step()  # 1 if |x_i-y_j| <= e, 0 otherwise
        volumes = float( np.pi ** (self.D / 2) / gamma(self.D / 2 + 1) ) * (scales ** self.D)
        neighbors_ij = neighbors_ij / volumes

        if log_weights is None:
            neighbors_ij = neighbors_ij / len(source)
        else:
            w_j = LazyTensor(log_weights[None,:,None].exp())
            neighbors_ij = neighbors_ij * w_j
        
        densities_i = neighbors_ij.sum(axis=1)  # (N,K)

        return - (densities_i * probas[None,:]).sum(dim=1).log().view(-1)








class GaussianProposal(Proposal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    
    def sample_noise(self, N, scales):
        return scales * torch.randn(N, self.D).type(self.dtype)

    
    def nlog_density(self, target, source, log_weights, scales, probas):

        D_ij = squared_distances(target, source)
        logK_ij = - D_ij / (2 * scales**2) - (self.D/2) * float(np.log(2*np.pi)) - self.D * scales.log()
        
        if log_weights is None:
            logK_ij = logK_ij - float(np.log(len(source)))
        else:
            logW_j = LazyTensor(log_weights[None,:,None])
            logK_ij = logK_ij + logW_j
        
        logdensities_i = logK_ij.logsumexp(dim=1).reshape(-1)  # (N,K)
        return - (logdensities_i + probas.log()[None,:]).logsumexp(dim=1).view(-1)

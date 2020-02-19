"""
Sampling on the 3D rotation group
===================================

Blabla

"""

######################
# Blabla

import numpy as np
import torch
from matplotlib import pyplot as plt

import warnings
import matplotlib.cbook

warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
plt.rcParams.update({'figure.max_open_warning': 0})

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

################################
# Blabla

from monaco.rotations import Rotations

space = Rotations(dtype = dtype)


#######################################
#

N = 10000 if use_cuda else 50

ref = torch.ones(N,1).type(dtype) * torch.FloatTensor([1, 0, 0, 0]).type(dtype)
ref2 = torch.ones(N,1).type(dtype) * torch.FloatTensor([1, 5, 5, 0]).type(dtype)

ref = torch.cat((ref, ref2), dim=0)

from monaco.rotations import BallProposal

proposal = BallProposal(space, scale = .5)

x = proposal.sample(ref)

# Display the initial configuration:
plt.figure(figsize = (8, 8))
space.scatter( x, "red" )
space.draw_frame()
plt.tight_layout()


#######################################
#

from monaco.rotations import RejectionSampling, quat_to_matrices

J = 10 * torch.randn(3, 3).type(dtype)

J = torch.FloatTensor([[1, 5, 5, 1]]).type(dtype)
J = 50 * quat_to_matrices(J)

def von_mises_potential(x):
    A = quat_to_matrices(x)
    V_i = - .5 * (J.view(-1, 9) * A.view(-1, 9)).sum(1)

    # u, s, v = torch.svd(J)
    # V_i = V_i + .5 * s.sum()

    return V_i


distribution = RejectionSampling(space, von_mises_potential)

if False:
    x = distribution.sample(N)

    # Display the initial configuration:
    plt.figure(figsize = (8, 8))
    space.scatter( x, "red" )
    space.plot( distribution.potential, "red")
    space.draw_frame()




##########################################
#

from monaco.samplers import CMC, display_samples


start = space.uniform_sample(N)
cmc_sampler = CMC(space, start, proposal).fit(distribution)
# display_samples(cmc_sampler, iterations = 100, runs = 5)









####################################
#


from monaco.rotations import quat_to_matrices

class ProcrustesDistribution(object):
    def __init__(self, source, target, temperature = 1.):
        self.source = source
        self.target = target
        self.temperature = temperature

    def potential(self, q):
        """Evaluates the potential on the point cloud x."""
        R = quat_to_matrices(q)  # (N, 3, 3)
        models = R @ self.source.t()  # (N, 3, npoints)

        V_i = ((models - self.target.t().view(1,3,-1))**2).mean(2).sum(1)

        return V_i.view(-1) / (2 * self.temperature)  # (N,)


#########################################
#

def load_csv(fname):
    x = np.loadtxt(fname, skiprows = 1, delimiter = ',')
    x = torch.from_numpy(x).type(dtype)
    x -= x.mean(0)
    scale = (x**2).sum(1).mean(0)
    x /= scale
    return x

A = load_csv("data/Ca1UBQ.csv")
B = load_csv("data/Ca1D3Z_1.csv")

distribution = ProcrustesDistribution(A, B, temperature = 1e-4)

#########################################
#


from monaco.samplers import MOKA_CMC, display_samples

N = 10000 if use_cuda else 50

start = space.uniform_sample(N)
proposal = BallProposal(space, scale = [.1, .2, .5, 1., 2.])

moka_sampler = MOKA_CMC(space, start, proposal, annealing = 5).fit(distribution)
display_samples(moka_sampler, iterations = 100, runs = 2);




##################################################
#

N = 10000

from monaco.rotations import quat_to_matrices
from geomloss import SamplesLoss

wasserstein = SamplesLoss("sinkhorn", p=2, blur=.05)


class WassersteinDistribution(object):
    def __init__(self, source, target, temperature = 1.):
        self.source = source
        self.target = target
        self.temperature = temperature

    def potential(self, q):
        """Evaluates the potential on the point cloud x."""
        R = quat_to_matrices(q)  # (N, 3, 3)
        models = R @ self.source.t()  # (N, 3, npoints)
        
        N = len(models)
        models = models.permute(0,2,1).contiguous()
        targets = self.target.repeat(N, 1, 1).contiguous()
        
        V_i = wasserstein(models, targets)

        return V_i.view(-1) / self.temperature  # (N,)



distribution = WassersteinDistribution(A, B, temperature = 1e-4)

start = space.uniform_sample(N)
proposal = BallProposal(space, scale = [.1, .2, .5, 1., 2.])

moka_sampler = MOKA_CMC(space, start, proposal, annealing = 5).fit(distribution)
display_samples(moka_sampler, iterations = 100, runs = 1);




#############################################
#

def load_coordinates(coordinates):
    x = torch.FloatTensor(coordinates).type(dtype)
    x -= x.mean(0)
    scale = (x**2).sum(1).mean(0)
    x /= scale
    return x

A = load_coordinates([[1., 0., 0.], [-1., 0., 0.]])
B = load_coordinates([[0., 1., 0.], [0., -1., 0.]])

distribution = WassersteinDistribution(A, B, temperature = 1e-4)


start = space.uniform_sample(N)
proposal = BallProposal(space, scale = [.1, .2, .5, 1., 2.])

moka_sampler = MOKA_CMC(space, start, proposal, annealing = 5).fit(distribution)
display_samples(moka_sampler, iterations = 100, runs = 2)


plt.show()
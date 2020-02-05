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

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

################################
# Blabla

from monaco.rotations import Rotations

space = Rotations(dtype = dtype)


#######################################
#

N = 2000 if use_cuda else 50

ref = torch.ones(N,1).type(dtype) * torch.FloatTensor([1, 0, 0, 0]).type(dtype)
ref2 = torch.ones(N,1).type(dtype) * torch.FloatTensor([1, 5, 5, 0]).type(dtype)

ref = torch.cat((ref, ref2), dim=0)

from monaco.rotations import BallProposal

proposal = BallProposal(space, scale = .1)

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
display_samples(cmc_sampler, iterations = 100, runs = 5)






####################################
#


from monaco.rotations import quat_to_matrices

class ProcrustesDistribution(object):
    def __init__(self, source, target, strength = 1.):
        self.source = source
        self.target = target
        self.strength = strength

    def potential(self, q):
        """Evaluates the potential on the point cloud x."""
        R = quat_to_matrices(q)  # (N, 3, 3)
        models = R @ self.source.t()  # (N, 3, npoints)

        V_i = ((models - self.target.t().view(1,3,-1))**2).mean(2).sum(1)
        print(V_i.mean().item())

        return self.strength * V_i.view(-1)  # (N,)


def load_csv(fname):
    x = np.loadtxt(fname, skiprows = 1, delimiter = ',')
    x = torch.from_numpy(x).type(dtype)
    x -= x.mean(0)
    scale = (x**2).sum(1).mean(0)
    x /= scale
    return x

A = load_csv("data/Ca1UBQ.csv")
B = load_csv("data/Ca1D3Z_1.csv")

distribution = ProcrustesDistribution(A, B)

#########################################
#

from monaco.samplers import CMC, display_samples

N = 10000 if use_cuda else 50

start = space.uniform_sample(N)
proposal = BallProposal(space, scale = .1)

cmc_sampler = CMC(space, start, proposal, annealing = 10).fit(distribution)
display_samples(cmc_sampler, iterations = 100, runs = 5)

plt.show()

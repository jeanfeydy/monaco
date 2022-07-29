"""
Sampling on the 3D rotation group
===================================

Let's show how to sample some distributions on the compact manifold SO(3).

"""

######################
# Introduction
# -----------------
#
# First, some standard imports.

import numpy as np
import torch
from matplotlib import pyplot as plt

import warnings
import matplotlib.cbook

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
plt.rcParams.update({"figure.max_open_warning": 0})

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

################################
# All functions that are relevant to the rotation group
# are stored in the `monaco.rotations` submodule.
# We first create our manifold, on the GPU whenever possible.

from monaco.rotations import Rotations

space = Rotations(dtype=dtype)


#######################################
# Rotations are encoded as quaternions and displayed as
# Euler vectors in a sphere of radius pi.
# Antipodal points at the boundary are identified with each other.
# Here, we create two arbitrary rotations and sample points
# at random in their neighborhood.

N = 10000 if use_cuda else 50

ref = torch.ones(N, 1).type(dtype) * torch.FloatTensor([1, 0, 0, 0]).type(dtype)
ref2 = torch.ones(N, 1).type(dtype) * torch.FloatTensor([1, 5, 5, 0]).type(dtype)

ref = torch.cat((ref, ref2), dim=0)

from monaco.rotations import BallProposal

proposal = BallProposal(space, scale=0.5)

x = proposal.sample(ref)

# Display the initial configuration:
plt.figure(figsize=(8, 8))
space.scatter(x, "red")
space.draw_frame()
plt.tight_layout()


####################################
# Procrustes analysis
# ---------------------------
#
# We consider the Von Mises distribution
# associated to a Procrustes registration problem:


from monaco.rotations import quat_to_matrices


class ProcrustesDistribution(object):
    def __init__(self, source, target, temperature=1.0):
        self.source = source
        self.target = target
        self.temperature = temperature

    def potential(self, q):
        """Evaluates the potential on the point cloud x."""
        R = quat_to_matrices(q)  # (N, 3, 3)
        models = R @ self.source.t()  # (N, 3, npoints)

        V_i = ((models - self.target.t().view(1, 3, -1)) ** 2).mean(2).sum(1)

        return V_i.view(-1) / (2 * self.temperature)  # (N,)


#########################################
# Then, we load two proteins as point clouds in the ambient 3D space:
#


def load_csv(fname):
    x = np.loadtxt(fname, skiprows=1, delimiter=",")
    x = torch.from_numpy(x).type(dtype)
    x -= x.mean(0)
    scale = (x**2).sum(1).mean(0)
    x /= scale
    return x


A = load_csv("data/Ca1UBQ.csv")
B = load_csv("data/Ca1D3Z_1.csv")

distribution = ProcrustesDistribution(A, B, temperature=1e-4)

#########################################
# Finally, we use the MOKA sampler on this distribution:
#


from monaco.samplers import MOKA_CMC, display_samples

N = 10000 if use_cuda else 50

start = space.uniform_sample(N)
proposal = BallProposal(space, scale=[0.1, 0.2, 0.5, 1.0, 2.0])

moka_sampler = MOKA_CMC(space, start, proposal, annealing=5).fit(distribution)
display_samples(moka_sampler, iterations=100, runs=2, small=False)


##################################################
# Wasserstein potential
# ------------------------------
#
# We rely on the GeomLoss library to define a Procrustes-like
# potential, where the discrepancy between two point clouds is computed
# using a good approximation of the squared Wasserstein distance.
#

N = 10000

from monaco.rotations import quat_to_matrices
from geomloss import SamplesLoss

wasserstein = SamplesLoss("sinkhorn", p=2, blur=0.05)


class WassersteinDistribution(object):
    def __init__(self, source, target, temperature=1.0):
        self.source = source
        self.target = target
        self.temperature = temperature

    def potential(self, q):
        """Evaluates the potential on the point cloud x."""
        R = quat_to_matrices(q)  # (N, 3, 3)
        models = R @ self.source.t()  # (N, 3, npoints)

        N = len(models)
        models = models.permute(0, 2, 1).contiguous()
        targets = self.target.repeat(N, 1, 1).contiguous()

        V_i = wasserstein(models, targets)

        return V_i.view(-1) / self.temperature  # (N,)


####################################################
# Just as in the example above, we use
# the MOKA algorithm to generate samples for this
# useful distribution:

distribution = WassersteinDistribution(A, B, temperature=1e-4)

start = space.uniform_sample(N)
proposal = BallProposal(space, scale=[0.1, 0.2, 0.5, 1.0, 2.0])

moka_sampler = MOKA_CMC(space, start, proposal, annealing=5).fit(distribution)
display_samples(moka_sampler, iterations=100, runs=1, small=False)


#############################################
# As a sanity check, we perform the same computation
# with simple pairs of points.
#


def load_coordinates(coordinates):
    x = torch.FloatTensor(coordinates).type(dtype)
    x -= x.mean(0)
    scale = (x**2).sum(1).mean(0)
    x /= scale
    return x


A = load_coordinates([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
B = load_coordinates([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])

distribution = WassersteinDistribution(A, B, temperature=1e-4)

############################################
# As expected, all the symmetries of the problem are respected by our sampler:

start = space.uniform_sample(N)
proposal = BallProposal(space, scale=[0.1, 0.2, 0.5, 1.0, 2.0])

moka_sampler = MOKA_CMC(space, start, proposal, annealing=5).fit(distribution)
display_samples(moka_sampler, iterations=100, runs=2, small=False)


plt.show()

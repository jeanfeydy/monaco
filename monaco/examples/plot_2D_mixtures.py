"""
Sampling in 2D
===============================

We discuss the performances of several Monte Carlo samplers on a toy 2D example.

"""


######################
# Introduction
# -------------------
#
# First of all, some standard imports.


import numpy as np
import torch
from matplotlib import pyplot as plt

plt.rcParams.update({"figure.max_open_warning": 0})

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

################################
# Our sampling space:

from monaco.euclidean import EuclideanSpace

D = 2
space = EuclideanSpace(dimension=D, dtype=dtype)


#######################################
# Our toy target distribution:


from monaco.euclidean import UnitPotential

N, M = (10000 if use_cuda else 50), 5
nruns = 5


def sinc_potential(x, stripes=3):
    sqnorm = (x**2).sum(-1)
    V_i = np.pi * stripes * sqnorm
    V_i = (V_i.sin() / V_i) ** 2
    return -V_i.log()


distribution = UnitPotential(space, sinc_potential)


#############################
# Display the target density, with a typical sample.

plt.figure(figsize=(8, 8))
space.scatter(distribution.sample(N), "red")
space.plot(distribution.potential, "red")
space.draw_frame()


########################################
# Sampling
# --------------------
#
# We start from a very poor initialization,
# thus simulating the challenge of sampling an unknown distribution.

start = 0.9 + 0.1 * torch.rand(N, D).type(dtype)


#########################################
# For exploration, we generate a fraction of our samples
# using a simple uniform distribution.

from monaco.euclidean import UniformProposal

exploration = 0.05
exploration_proposal = UniformProposal(space)


#######################################
# Our proposal will stay the same throughout the experiments:
# a combination of uniform samples on balls with radii that
# range from 1/1000 to  0.3.

from monaco.euclidean import BallProposal

proposal = BallProposal(
    space,
    scale=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
    exploration=exploration,
    exploration_proposal=exploration_proposal,
)


##########################################
# First of all, we illustrate a run of the standard
# Metropolis-Hastings algorithm, parallelized on the GPU:


info = {}

from monaco.samplers import ParallelMetropolisHastings, display_samples

pmh_sampler = ParallelMetropolisHastings(space, start, proposal, annealing=None).fit(
    distribution
)
info["PMH"] = display_samples(pmh_sampler, iterations=20, runs=nruns)


########################################
# Then, the standard Collective Monte Carlo method:


from monaco.samplers import CMC

proposal = BallProposal(
    space,
    scale=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
    exploration=exploration,
    exploration_proposal=exploration_proposal,
)

cmc_sampler = CMC(space, start, proposal, annealing=None).fit(distribution)
info["CMC"] = display_samples(cmc_sampler, iterations=20, runs=nruns)


########################################
# BGK - Collective Monte Carlo method:

from monaco.samplers import Ada_CMC
from monaco.euclidean import GaussianProposal

gaussian_proposal = GaussianProposal(
    space,
    scale=[0.1],
    exploration=exploration,
    exploration_proposal=exploration_proposal,
)
bgk_sampler = Ada_CMC(space, start, gaussian_proposal, annealing=5).fit(distribution)
info["BGK_CMC"] = display_samples(bgk_sampler, iterations=20, runs=1)


########################################
# GMM - Collective Monte Carlo method:

from monaco.euclidean import GMMProposal

gmm_proposal = GMMProposal(
    space,
    n_classes=100,
    exploration=exploration,
    exploration_proposal=exploration_proposal,
)
gmm_sampler = Ada_CMC(space, start, gmm_proposal, annealing=5).fit(distribution)
# info["GMM_CMC"] = display_samples(gmm_sampler, iterations=20, runs=1)


#############################
# With a Markovian selection of the kernel bandwidth:

from monaco.samplers import MOKA_Markov_CMC

proposal = BallProposal(
    space,
    scale=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
    exploration=exploration,
    exploration_proposal=exploration_proposal,
)

moka_markov_sampler = MOKA_Markov_CMC(space, start, proposal, annealing=5).fit(
    distribution
)
info["MOKA Markov"] = display_samples(moka_markov_sampler, iterations=20, runs=nruns)


#############################
# CMC with Richardson-Lucy deconvolution:

from monaco.samplers import KIDS_CMC

proposal = BallProposal(
    space,
    scale=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
    exploration=exploration,
    exploration_proposal=exploration_proposal,
)

kids_sampler = KIDS_CMC(space, start, proposal, annealing=None, iterations=50).fit(
    distribution
)
info["KIDS"] = display_samples(kids_sampler, iterations=20, runs=nruns)


#############################
# Finally, the Non Parametric Adaptive Importance Sampler,
# an efficient non-Markovian method with an extensive
# memory usage:


from monaco.samplers import SAIS

proposal = BallProposal(
    space, scale=0.1, exploration=exploration, exploration_proposal=exploration_proposal
)


class Q_0(object):
    def __init__(self):
        None

    def sample(self, n):
        return 0.9 + 0.1 * torch.rand(n, D).type(dtype)

    def potential(self, x):
        v = 100000 * torch.ones(len(x), 1).type_as(x)
        v[(x - 0.95).abs().max(1)[0] < 0.05] = -np.log(1 / 0.1)
        return v.view(-1)


q0 = Q_0()

sais_sampler = SAIS(space, start, proposal, annealing=None, q0=q0, N=N).fit(
    distribution
)
info["SAIS"] = display_samples(sais_sampler, iterations=20, runs=nruns)


###############################################
# Comparative benchmark:

import itertools
import seaborn as sns

iters = info["PMH"]["iteration"]


def display_line(key, marker):
    sns.lineplot(
        x=info[key]["iteration"],
        y=info[key]["error"],
        label=key,
        marker=marker,
        markersize=6,
        ci="sd",
    )


plt.figure(figsize=(4, 4))
markers = itertools.cycle(("o", "X", "P", "D", "^", "<", "v", ">", "*"))

for key, marker in zip(["PMH", "CMC", "MOKA Markov", "KIDS", "SAIS"], markers):
    display_line(key, marker)


plt.xlabel("Iterations")
plt.ylabel("ED ( sample, true distribution )")
plt.ylim(bottom=1e-4)
plt.yscale("log")

plt.tight_layout()


plt.show()

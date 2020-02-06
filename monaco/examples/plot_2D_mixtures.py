"""
Sampling in dimension 2
===============================

Blabla

"""

######################
# Blabla

import numpy as np
import torch
from matplotlib import pyplot as plt

plt.rcParams.update({'figure.max_open_warning': 0})

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

################################
# Blabla

from monaco.euclidean import EuclideanSpace

D = 2
space = EuclideanSpace(dimension = D, dtype = dtype)


#######################################
#

from monaco.euclidean import GaussianMixture

nruns = 2
N, M = (2000 if use_cuda else 50), 5

# Let's generate a blend of peaky Gaussians, in the unit square:
m = torch.rand(M, D).type(dtype)  # mean
s = torch.rand(M).type(dtype)     # deviation
w = torch.rand(M).type(dtype)     # weights

m = .25  + .5 * m
s = .005 + .1 * (s**6)
w = w / w.sum()  # normalize weights

distribution = GaussianMixture(space, m, s, w)


#######################################
#


from monaco.euclidean import UnitPotential

def sinc_potential(x, stripes = 3):
    sqnorm = (x**2).sum(-1)
    V_i = np.pi * stripes * sqnorm
    V_i = (V_i.sin() / V_i) ** 2
    return - V_i.log()

distribution = UnitPotential(space, sinc_potential)


# Display the initial configuration:
plt.figure(figsize = (8, 8))
space.scatter( distribution.sample(N), "red" )
space.plot( distribution.potential, "red")
space.draw_frame()



########################################
#

start = .9 + .1 * torch.rand(N, D).type(dtype)

from monaco.euclidean import BallProposal

proposal = BallProposal(space, scale = .1)

##########################################
#



info = {}

from monaco.samplers import ParallelMetropolisHastings, display_samples

pmh_sampler = ParallelMetropolisHastings(space, start, proposal, annealing = None).fit(distribution)
info["PMH"] = display_samples(pmh_sampler, iterations = 20, runs = nruns)


########################################
#

from monaco.samplers import CMC

cmc_sampler = CMC(space, start, proposal, annealing = None).fit(distribution)
info["CMC"] = display_samples(cmc_sampler, iterations = 20, runs = nruns)


#############################
#

from monaco.samplers import KIDS_CMC

kids_sampler = KIDS_CMC(space, start, proposal, annealing = None, iterations = 50).fit(distribution)
info["KIDS"] = display_samples(kids_sampler, iterations = 20, runs = nruns)


#############################
#

from monaco.samplers import NPAIS

proposal = BallProposal(space, scale = .1)

class Q_0(object):
    def __init__(self):
        None
    
    def sample(self, n):
        return .9 + .1 * torch.rand(n, D).type(dtype)

    def potential(self, x):
        v = 100000 * torch.ones(len(x), 1).type_as(x)
        v[(x - .95).abs().max(1)[0] < .05]  = - np.log(1 / .1)
        return v.view(-1)

q0 = Q_0()

npais_sampler = NPAIS(space, start, proposal, annealing = None, q0 = q0, N = N).fit(distribution)
info["NPAIS"] = display_samples(npais_sampler, iterations = 20, runs = nruns)




import itertools
import seaborn as sns

iters = info["PMH"]["iteration"]

def display_line(key, marker):
    sns.lineplot(x = info[key]["iteration"], y = info[key]["error"], label=key, 
                 marker = marker, markersize = 6, ci="sd")

plt.figure(figsize=(4,4))
markers = itertools.cycle(('o', 'X', 'P', 'D', '^', '<', 'v', '>', '*')) 

for key, marker in zip(["PMH", "CMC", "KIDS", "NPAIS"], markers):
    display_line(key, marker)


plt.xlabel("Iterations")
plt.ylabel("ED ( sample, true distribution )")
plt.ylim(bottom = .001)
plt.yscale("log")

plt.tight_layout()



plt.show()
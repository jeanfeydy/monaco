"""
Sampling in dimension 1
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

D = 1
space = EuclideanSpace(dimension = D, dtype = dtype)


#######################################
#

from monaco.euclidean import GaussianMixture, UnitPotential

N, M = (10000 if use_cuda else 50), 5
Nlucky = (10 if use_cuda else 2)
nruns = 5

test_case = "sophia"

if test_case == "gaussians":
    # Let's generate a blend of peaky Gaussians, in the unit square:
    m = torch.rand(M, D).type(dtype)  # mean
    s = torch.rand(M).type(dtype)     # deviation
    w = torch.rand(M).type(dtype)     # weights

    m = .25  + .5 * m
    s = .005 + .1 * (s**6)
    w = w / w.sum()  # normalize weights

    distribution = GaussianMixture(space, m, s, w)


elif test_case == "sophia":
    m = torch.FloatTensor([.5,    .1,   .2,   .8,  .9 ]).type(dtype)[:,None]
    s = torch.FloatTensor([.15, .005, .002,  .002, .005]).type(dtype)
    w = torch.FloatTensor([.1,  2/12,  1/12, 1/12, 2/12]).type(dtype)
    w = w / w.sum()  # normalize weights

    distribution = GaussianMixture(space, m, s, w)

elif test_case == "ackley":

    def ackley_potential(x, stripes = 15):
        f_1 = 20 * (-.2 * (((x - .5) * stripes)**2).mean(-1).sqrt()).exp()
        f_2 = ((2 * np.pi * ((x-.5) * stripes ) ).cos().mean(-1)).exp()

        return - (f_1 + f_2 - np.exp(1) - 20) / stripes

    distribution = UnitPotential(space, ackley_potential)



# Display the initial configuration:
plt.figure(figsize = (8, 8))
space.scatter( distribution.sample(N), "red" )
space.plot( distribution.potential, "red")
space.draw_frame()
#plt.show()


start = .05 + .1 * torch.rand(N, D).type(dtype)
start[:Nlucky] = .9

#######################################
#

from monaco.euclidean import BallProposal

proposal = BallProposal(space, scale = .05)

##########################################
#


from monaco.samplers import ParallelMetropolisHastings, display_samples

pmh_sampler = ParallelMetropolisHastings(space, start, proposal).fit(distribution)
display_samples(pmh_sampler, iterations = 20, runs = nruns)



########################################
#


from monaco.samplers import CMC
cmc_sampler = CMC(space, start, proposal).fit(distribution)
display_samples(cmc_sampler, iterations = 20, runs = nruns)

#######################
#


acmc_sampler = CMC(space, start, proposal, annealing = 5).fit(distribution)
display_samples(acmc_sampler, iterations = 20, runs = nruns)


#############################
#

if False:
    from monaco.samplers import KIDS_CMC

    kids_sampler = KIDS_CMC(space, start, proposal, annealing = 10, iterations = 50).fit(distribution)
    display_samples(kids_sampler, iterations = 20, runs = nruns)



#############################
#

from monaco.samplers import MOKA_CMC

proposal = BallProposal(space, scale = [.001, .003, .01, .03, .1, .3])

moka_sampler = MOKA_CMC(space, start, proposal, annealing = 10).fit(distribution)
display_samples(moka_sampler, iterations = 20, runs = nruns)


plt.show()
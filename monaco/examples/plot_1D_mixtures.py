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

N, M = (2000 if use_cuda else 50), 5

test_case = "ackley"

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
    w = torch.FloatTensor([.5,  2/12,  1/12, 1/12, 2/12]).type(dtype)
    w = w / w.sum()  # normalize weights

    distribution = GaussianMixture(space, m, s, w)

elif test_case == "ackley":

    def ackley_potential(x, stripes = 15):
        f_1 = 20 * (-.2 * (((x - .5) * stripes)**2).mean(-1).sqrt()).exp()
        f_2 = ((2 * np.pi * ((x-.5) * stripes ) ).cos().mean(-1)).exp()

        return - (f_1 + f_2 - np.exp(1) - 20) / stripes

    distribution = UnitPotential(space, ackley_potential)

print(ackley_potential(torch.FloatTensor([.5])))



# Display the initial configuration:
plt.figure(figsize = (8, 8))
space.scatter( distribution.sample(N), "red" )
space.plot( distribution.potential, "red")
space.draw_frame()
#plt.show()


#######################################
#

from monaco.euclidean import BallProposal

proposal = BallProposal(space, scale = .05)

########################################
#


from monaco.samplers import CMC
start = .1 + .1 * torch.rand(N, D).type(dtype)
cmc_sampler = CMC(space, start, proposal).fit(distribution)

###########################################
#

from monaco.samplers import display_samples
#display_samples(cmc_sampler, iterations = 100)

#######################
#


start = .1 + .1 * torch.rand(N, D).type(dtype)
acmc_sampler = CMC(space, start, proposal, annealing = 10).fit(distribution)
#display_samples(acmc_sampler, iterations = 100)

#############################
#

from monaco.samplers import MOKA_CMC

proposal = BallProposal(space, scale = [.001, .002, .005, .01, .02, .05, .1, .2, .5])

start = .1 + .1 * torch.rand(N, D).type(dtype)
moka_sampler = MOKA_CMC(space, start, proposal, annealing = 10).fit(distribution)
display_samples(moka_sampler, iterations = 100)



#############################
#

if False:

    from monaco.samplers import KIDS_CMC

    start = .1 + .1 * torch.rand(N, D).type(dtype)
    kids_sampler = KIDS_CMC(space, start, proposal, annealing = 10, iterations = 50).fit(distribution)
    display_samples(kids_sampler, iterations = 100)

plt.show()
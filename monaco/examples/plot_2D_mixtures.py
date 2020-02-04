"""
Sampling in dimension 2
===============================

Blabla

"""

######################
# Blabla

import torch
from matplotlib import pyplot as plt

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

N, M = (2000 if use_cuda else 50), 5

# Let's generate a blend of peaky Gaussians, in the unit square:
m = torch.rand(M, D).type(dtype)  # mean
s = torch.rand(M).type(dtype)     # deviation
w = torch.rand(M).type(dtype)     # weights

m = .25  + .5 * m
s = .005 + .1 * (s**6)
w = w / w.sum()  # normalize weights

distribution = GaussianMixture(space, m, s, w)


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

start = torch.rand(N, D).type(dtype)

##########################################
#

from monaco.samplers import CMC
cmc_sampler = CMC(space, start, proposal).fit(distribution)

###########################################
#

from monaco.samplers import display_samples
display_samples(cmc_sampler, iterations = 100)

#######################
#

start = torch.rand(N, D).type(dtype)
acmc_sampler = CMC(space, start, proposal, annealing = 10).fit(distribution)
display_samples(acmc_sampler, iterations = 100)

#############################
#

from monaco.samplers import KIDS_CMC

start = torch.rand(N, D).type(dtype)
kids_sampler = KIDS_CMC(space, start, proposal, annealing = 10, iterations = 5).fit(distribution)
display_samples(kids_sampler, iterations = 100)

plt.show()
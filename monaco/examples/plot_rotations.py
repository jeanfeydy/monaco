"""
Sampling on the 3D rotation group
===================================

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

from monaco.rotations import Rotations

space = Rotations(dtype = dtype)


#######################################
#

N = 2000 if use_cuda else 50

ref = torch.ones(N,1).type(dtype) * torch.FloatTensor([1, 0, 0, 0]).type(dtype)
ref2 = torch.ones(N,1).type(dtype) * torch.FloatTensor([1, 5, 5, 0]).type(dtype)

ref = torch.cat((ref, ref2), dim=0)

from monaco.rotations import BallProposal

proposal = BallProposal(space, scale = 1.)

x = proposal.sample(ref)

# Display the initial configuration:
plt.figure(figsize = (8, 8))
space.scatter( x, "red" )
space.draw_frame()
plt.tight_layout()

l = proposal.potential(ref[:N])(x)



#######################################
#

from monaco.rotations import RejectionSampling, quat_to_matrices

J = 10 * torch.randn(3, 3).type(dtype)

def von_mises_potential(x):
    A = quat_to_matrices(x)
    V_i = - .5 * (J.view(-1, 9) * A.view(-1, 9)).sum(1)

    u, s, v = torch.svd(J)
    V_i = V_i + .5 * s.sum()

    print(V_i.max(), V_i.min())
    return V_i


distribution = RejectionSampling(space, von_mises_potential)
x = distribution.sample(N)

# Display the initial configuration:
plt.figure(figsize = (8, 8))
space.scatter( x, "red" )
space.plot( distribution.potential, "red")
space.draw_frame()




##########################################
#

from monaco.samplers import CMC, display_samples

cmc_sampler = CMC(space, x, proposal).fit(distribution)
display_samples(cmc_sampler, iterations = 100, runs = 5)



plt.show()
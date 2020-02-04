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

N = 100
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

plt.show()
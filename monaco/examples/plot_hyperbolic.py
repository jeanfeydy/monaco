"""
Sampling on the Poincaré disk
===============================

Blabla

"""

######################
# Blabla


import torch

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import umap

numpy = lambda x : x.cpu().numpy()


try:
    embedding = np.load("data/hyperbolic_embedding.npy")
    disk_x, disk_y = embedding[:,0], embedding[:,1]

    labels = np.load("data/hyperbolic_labels.npy")

except IOError:
    dataset = sklearn.datasets.fetch_openml('mnist_784')

    print(dataset.data.shape)

    features, labels = dataset.data[:10000], dataset.target[:10000].astype('int64')

    hyperbolic_mapper = umap.UMAP(target_metric='hyperboloid',
                                random_state=42).fit(features)

    x = hyperbolic_mapper.embedding_[:, 0]
    y = hyperbolic_mapper.embedding_[:, 1]
    z = np.sqrt(1 + np.sum(hyperbolic_mapper.embedding_**2, axis=1))

    disk_x = x / (1 + z)
    disk_y = y / (1 + z)
    embedding = np.stack((disk_x, disk_y)).T

    np.save("data/hyperbolic_embedding.npy", embedding)
    np.save("data/hyperbolic_labels.npy", labels)


fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.scatter(disk_x, disk_y, c=labels, s=2000/len(labels), cmap='Spectral')
boundary = plt.Circle((0,0), 1, fc='none', ec='k')
ax.add_artist(boundary)
plt.axis("equal")
plt.axis([-1.1,1.1,-1.1,1.1])
ax.axis('off');


##########################################
#

from monaco.hyperbolic import HyperbolicSpace, disk_to_halfplane

space = HyperbolicSpace(dimension = 2, dtype = dtype)

X = torch.from_numpy(embedding).type(dtype)
X = disk_to_halfplane(X)

fig = plt.figure(figsize=(8,8))

plt.scatter(numpy(X)[:,0], numpy(X)[:,1], c = labels, s = 2000 / len(X), cmap='Spectral')
plt.yscale('log')



fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)

space.scatter(X, "red")
space.draw_frame()




####################################
#

from monaco.hyperbolic import BallProposal

proposal = BallProposal(space, scale = 1.75)

A = torch.FloatTensor([0, 1]).type(dtype)
B = torch.FloatTensor([4,.5]).type(dtype)

ref = torch.stack( (A,)*1000 + (B,)*1000, dim=0)

d_AB = 1 + ((A - B)**2).sum() / (2 * A[1] * B[1])
d_AB = (d_AB + (d_AB**2 - 1).sqrt()).log()
print(d_AB)


from monaco.hyperbolic import halfplane_to_disk

print(halfplane_to_disk(A))
C = halfplane_to_disk(B)
print(C)
R = (C ** 2).sum().sqrt()
d_IC = ((1 + R) / (1 - R)).log()
print(d_IC)


x = proposal.sample(ref)


# Display the initial configuration:
plt.figure(figsize = (8, 8))
space.scatter( x, "red" )
space.draw_frame()
plt.tight_layout()



fig = plt.figure(figsize=(8,8))
plt.scatter(numpy(x)[:,0], numpy(x)[:,1], c = "red", s = 2000 / len(x))
plt.yscale('log')




####################################
#


from pykeops.torch import LazyTensor

class DistanceDistribution(object):
    def __init__(self, points):
        self.points = points

    def potential(self, x):
        """Evaluates the potential on the point cloud x."""

        x_i = LazyTensor(x[:,None,:])
        y_j = LazyTensor(self.points[None,:,:])

        D_ij = ((x_i - y_j)**2).sum(-1)
        D_ij = 1 + D_ij / (2 * x_i[1] * y_j[1])
        D_ij = (D_ij + (D_ij**2 - 1).sqrt()).log()

        V_i = D_ij.min(dim=1)

        return V_i.reshape(-1)  # (N,)

target = X if use_cuda else X[:100]

distribution = DistanceDistribution(target)

#########################################
#

from monaco.samplers import CMC, display_samples

N = 1000 if use_cuda else 50
start = 1. + torch.rand(N, 2).type(dtype)

proposal = BallProposal(space, scale = 0.5)

cmc_sampler = CMC(space, start, proposal).fit(distribution)
display_samples(cmc_sampler, iterations = 100, runs = 5)

plt.show()

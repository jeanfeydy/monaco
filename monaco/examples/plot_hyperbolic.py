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

x = torch.from_numpy(embedding).type(dtype)
x = disk_to_halfplane(x)

numpy = lambda x : x.cpu().numpy()

fig = plt.figure(figsize=(8,8))

plt.scatter(numpy(x)[:,0], numpy(x)[:,1], c = labels, s = 2000 / len(x), cmap='Spectral')
plt.yscale('log')



fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)

space.scatter(x, "red")
space.draw_frame()

plt.show()


import numpy as np
import torch
import torch.nn.functional as F

from matplotlib import pyplot as plt
from pykeops.torch  import LazyTensor

from mpl_toolkits.mplot3d import Axes3D

from .proposals import Proposal

numpy = lambda x : x.cpu().numpy()

def normalize(points):
    return F.normalize(points, p=2, dim=1)


def halfplane_to_disk(points):
    """Moebius map from the Poincare half-plane to the Poincare disk."""
    x = points[...,:-1]
    y = points[...,-1:]
    sqnorms = (points ** 2).sum(-1, keepdim=True)
    s = 1 / (1 + 2 * y + sqnorms)

    return torch.cat( (2 * s * x, s * (sqnorms - 1)), dim=-1)


def disk_to_halfplane(points):
    """Moebius map from the Poincare half-plane to the Poincare disk."""
    x = points[...,:-1]
    y = points[...,-1:]
    sqnorms = (points ** 2).sum(-1, keepdim=True)
    s = 1 / (1 - 2 * y + sqnorms)

    return torch.cat( (2 * s * x, s * (1 - sqnorms)), dim=-1)




class HyperbolicSpace(object):
    """Hyperbolic space of dimension D.
    
    Points are encoded in the Poincare half-plane, but displayed on the disk.
    """

    def __init__(self, dimension = 2, dtype = None, resolution = 200):
        
        if dtype is None: dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.dtype = dtype
        self.dimension = dimension

        if self.dimension != 2:
            raise NotImplementedError()

        
        # Display: create a uniform grid on the square [-1,1]^2
        self.resolution = resolution
        ticks = np.linspace(-1, 1, self.resolution + 1)[:-1] + .5 / self.resolution
        x_g, y_g = np.meshgrid(ticks, ticks)
        self.grid = torch.from_numpy( np.vstack((x_g.ravel(), y_g.ravel())).T ).type(dtype).contiguous()

    
    def apply_noise(self, z, v):
        """Assumes that noise v is centered around i = (0, 1)."""
        w = z[:,-1:] * v.clone()
        w[:,:-1] += z[:,:-1]
        return w


    def discrepancy(self, a, b):
        """Computes the energy distance between two samples."""
        return 0.  # This feature is not used in the paper


    def scatter(self, points, color, ax = None):
        """Displays a sample as a point cloud on the Poincare disk."""

        if ax is None: ax = plt.gca()

        disk = numpy(halfplane_to_disk(points))
        ax.scatter(disk[:, 0], disk[:, 1], s = 2000 / len(disk), c = color)


    def plot(self, potential, color, ax = None):
        """Displays a potential as a contour plot on the Poincare disk."""

        if ax is None: ax = plt.gca()

        disk = self.grid.clone()
        mask = (disk**2).sum(1) < 1
        unit_disk = disk[mask, :].contiguous()
        halfplane = disk_to_halfplane(unit_disk)

        log_heatmap = torch.zeros(self.resolution ** 2).type(self.dtype)
        log_heatmap[:] = np.nan
        log_heatmap[mask] = potential(halfplane)

        scale  = np.amax(np.abs(numpy(log_heatmap[mask])))
        levels = np.linspace(-scale, scale, 41)

        log_heatmap = numpy(log_heatmap)
        log_heatmap = log_heatmap.reshape(self.resolution, self.resolution)

        ax.contour(log_heatmap, origin='lower', linewidths=1., colors=color,
                    levels=levels, extent=(-1, 1, -1, 1))

    
    def draw_frame(self, ax = None):
        """Draws the border of the Poincare disk."""

        if ax is None: ax = plt.gca()
            
        boundary = plt.Circle((0,0), 1, fc='none', ec='k')
        ax.add_artist(boundary)
        plt.axis("equal")
        plt.axis([-1.1,1.1,-1.1,1.1])
        ax.axis('off');





class BallProposal(Proposal):
    """Uniform proposal on a ball of the hyperbolic plane."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def sample_radii(self, N, scales):
        """Returns hyperbolic radii and angles."""

        if self.D == 2:
            y = torch.rand(N, 1).type(self.dtype)
            y = 1 + y * ((scales.exp() + (-scales).exp())/2 - 1) 
            return (y + (y**2 - 1).sqrt()).log()
        else:
            raise NotImplementedError()
            radii  = scales * torch.rand(N, 1).type(self.dtype)
            uniform = torch.rand(N, 1).type(self.dtype)

            threshold = (radii.exp() - (-radii).exp()) / (scales.exp() - (-scales).exp())
            reject = (uniform > threshold).view(-1)
            M = int(reject.sum())

            if M == 0:
                return angles
            else:
                angles[reject] = self.sample_angles(M, scales[reject])
                return angles


    def sample_noise(self, N, scales):
        """Samples noise on the Poincare half-plane."""

        radii = self.sample_radii(N, scales)

        disk_radii = ((radii).exp() - 1) / ((radii).exp() + 1)

        directions = torch.randn(N, 2).type(self.dtype)
        directions = normalize(directions)  # Direction, randomly sampled on the sphere

        disk_noise = disk_radii.view(-1,1) * directions.view(-1,2)
        return disk_to_halfplane(disk_noise)


    def nlog_density(self, target, source, log_weights, scales, probas):
        """Negative log-likelihood of the proposal generated by the source onto the target."""

        x_i = LazyTensor( target[:,None,:] )  # (N,1,D)
        y_j = LazyTensor( source[None,:,:] )  # (1,M,D)
        
        D_ij = ((x_i - y_j)**2).sum(-1)
        D_ij = 1 + D_ij / (2 * x_i[self.D - 1] * y_j[self.D - 1])
        D_ij = (D_ij + (D_ij**2 - 1).sqrt()).log()  # (N,M)
        
        neighbors_ij = (scales - D_ij).step()  # 1 if |x_i-y_j| <= scales, 0 otherwise

        if self.D == 2:
            volumes = float( np.pi ) * (scales.exp() + (-scales).exp())
        else:
            raise NotImplementedError()

        neighbors_ij = neighbors_ij / volumes

        if log_weights is None:
            neighbors_ij = neighbors_ij / len(source)
        else:
            w_j = LazyTensor(log_weights[None,:,None].exp())
            neighbors_ij = neighbors_ij * w_j
        
        densities_i = neighbors_ij.sum(axis=1)  # (N,K)

        return - (densities_i * probas[None,:]).sum(dim=1).log().view(-1)







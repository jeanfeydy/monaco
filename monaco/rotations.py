import numpy as np
import torch
import torch.nn.functional as F

from matplotlib import pyplot as plt
from pykeops.torch  import LazyTensor

from mpl_toolkits.mplot3d import Axes3D

from .proposals import Proposal

numpy = lambda x : x.cpu().numpy()



def squared_distances(x, y):
    x_i = LazyTensor( x[:,None,:] )  # (N,1,D)
    y_j = LazyTensor( y[None,:,:] )  # (1,M,D)
    return ((x_i - y_j)**2).sum(dim=2)  # (N,M,1)


def normalize(points):
    return F.normalize(points, p=2, dim=1)



def quat_mult(q_1, q_2):
    
    a_1, b_1, c_1, d_1 = q_1[:,0], q_1[:,1], q_1[:,2], q_1[:,3]
    a_2, b_2, c_2, d_2 = q_2[:,0], q_2[:,1], q_2[:,2], q_2[:,3]

    q_1_q_2 = torch.stack((
        a_1 * a_2 - b_1 * b_2 - c_1 * c_2 - d_1 * d_2,
        a_1 * b_2 + b_1 * a_2 + c_1 * d_2 - d_1 * c_2,
        a_1 * c_2 - b_1 * d_2 + c_1 * a_2 + d_1 * b_2,
        a_1 * d_2 + b_1 * c_2 - c_1 * b_2 + d_1 * a_2
        ), dim = 1)

    return q_1_q_2


def quat_to_matrices(points):
    points = normalize(points)
    w, x, y, z = points[:,0], points[:,1], points[:,2], points[:,3]
    Q = torch.stack(
            (1 - 2 * y**2 - 2 * z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w,
             2*x*y + 2*z*w, 1 - 2 * x**2 - 2 * z**2, 2*y*z - 2*x*w,
             2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2 * x**2 - 2 * y**2),
        dim = 1)
    return Q.view(-1,3,3)


def quat_to_angles_directions(points):
    points = normalize(points)
    norms = (points[:,1:]**2).sum(1).sqrt()
    angles = 2 * torch.atan2(norms, points[:,0])
    directions = points[:,1:] / norms.view(-1,1)

    angles[angles >  np.pi] -= 2 * np.pi
    angles[angles < -np.pi] += 2 * np.pi

    return angles, directions.contiguous()


def angles_directions_to_quat(angles, directions):
    t = angles / 2
    return torch.cat( ( t.cos().view(-1,1), 
                        t.sin().view(-1,1) * directions ),
                        dim = 1 )



class Rotations(object):

    def __init__(self, dtype = None):
        
        if dtype is None: dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.dtype = dtype
        self.dimension = 3

    
    def apply_noise(self, x, v):
        return quat_mult(x, v)


    def discrepancy(self, a, b):
        """Computes the energy distance between two samples."""

        x, y = quat_to_matrices(a).view(-1,9), quat_to_matrices(b).view(-1,9)

        n, m = len(x), len(y) 

        D_xx = squared_distances(x, x).sqrt().sum(dim=1).sum()
        D_xy = squared_distances(x, y).sqrt().sum(dim=1).sum()
        D_yy = squared_distances(y, y).sqrt().sum(dim=1).sum()

        return D_xy / (n*m) - .5 * (D_xx / (n*n) + D_yy / (m*m))


    def scatter(self, points, color, ax = None):

        if ax is None: ax = plt.gca()
        if type(ax) is not Axes3D:
            ax = plt.gcf().add_subplot(111, projection='3d')

        angles, directions = quat_to_angles_directions(points)
        euler = angles.view(-1,1) * directions
        angles, euler = numpy(angles), numpy(euler)

        ax.scatter(euler[:, 0], euler[:, 1], euler[:,2], 
                   s = 1000 / len(angles), c = angles)


    def plot(self, potential, color, ax = None):
        None

    
    def draw_frame(self, ax = None):
        if ax is None: ax = plt.gca()

        # Sphere of radius pi:
        theta = np.linspace(-np.pi, np.pi, 65)
        for z in np.linspace(-.9, .9, 7):
            r = np.sqrt(1 - z**2)
            x = r * np.sin(theta)
            y = r * np.cos(theta)
            ax.plot(np.pi * x, np.pi * y, np.pi * z, linewidth = .5, color = "k")

        for t in np.linspace(-np.pi, np.pi, 7):
            x = np.cos(t) * np.sin(theta)
            y = np.sin(t) * np.sin(theta)
            z = np.cos(theta)
            ax.plot(np.pi * x, np.pi * y, np.pi * z, linewidth = .5, color = "k")


        # Axis equal:             
        x_limits = [-1.1 * np.pi, 1.1 * np.pi] #ax.get_xlim3d()
        y_limits = [-1.1 * np.pi, 1.1 * np.pi] #ax.get_ylim3d()
        z_limits = [-1.1 * np.pi, 1.1 * np.pi] #ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

        ax.axis("off")
        ax.view_init(azim=-80, elev=30)

    
    def uniform_sample(self, N):
        uniform_sampler = BallProposal(self, [float(np.pi)])
        scales = np.pi * torch.ones(N,1).type(self.dtype)
        return uniform_sampler.sample_noise(N, scales)







class RejectionSampling(object):
    def __init__(self, space, potential):
        self.uniform_sampler = BallProposal(space, [np.pi])

        self.D = space.dimension
        self.dtype = space.dtype
        self.inner_potential = potential


    def potential(self, x):
        """Evaluates the potential on the point cloud x."""
        V_i = self.inner_potential(x)
        return V_i.reshape(-1)  # (N,)


    '''
    def sample(self, N = 1):
        """Returns a sample array of shape (N,D)."""
        ref = torch.ones(N,1).type(self.dtype) * torch.FloatTensor([1, 0, 0, 0]).type(self.dtype)
        return ref

        x = self.uniform_sampler.sample(ref).type(self.dtype)
        uniform = torch.rand(N).type(self.dtype)
        
        threshold = (- self.potential(x)).exp()
        reject = (uniform > threshold).view(-1)
        M = int(reject.sum())

        if M == 0:
            return x
        else:
            x[reject] = self.sample(M)
            return x
    '''







class BallProposal(Proposal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def sample_angles(self, N, scales):

        angles  = scales * torch.rand(N, 1).type(self.dtype)
        uniform = torch.rand(N, 1).type(self.dtype)

        threshold = (1 - angles.cos()) / (1 - scales.cos())
        reject = (uniform > threshold).view(-1)
        M = int(reject.sum())

        if M == 0:
            return angles
        else:
            angles[reject] = self.sample_angles(M, scales[reject])
            return angles


    def sample_noise(self, N, scales):
        angles = self.sample_angles(N, scales)
        
        directions = torch.randn(N, 3).type(self.dtype)
        directions = normalize(directions)  # Direction, randomly sampled on the sphere

        return angles_directions_to_quat(angles, directions)


    def nlog_density(self, target, source, log_weights, scales, logits):
        target, source = normalize(target), normalize(source)

        x_i = LazyTensor( target[:,None,:] )  # (N,1,D)
        y_j = LazyTensor( source[None,:,:] )  # (1,M,D)
        S_ij = (x_i | y_j).abs()  # (N,M)
        neighbors_ij = (S_ij - (scales / 2).cos()).step()  # 1 if |x_i-y_j| <= scales, 0 otherwise
        volumes = (scales - scales.sin()) / float( np.pi )
        neighbors_ij = neighbors_ij / volumes

        if log_weights is None:
            neighbors_ij = neighbors_ij / len(source)
        else:
            w_j = LazyTensor(log_weights[None,:,None].exp())
            neighbors_ij = neighbors_ij * w_j
        
        densities_i = neighbors_ij.sum(axis=1)  # (N,K)

        probas = logits.softmax(0)
        return - (densities_i * probas[None,:]).sum(dim=1).log().view(-1)







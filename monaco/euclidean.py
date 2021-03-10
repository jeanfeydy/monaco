import numpy as np
import torch
from matplotlib import pyplot as plt
from pykeops.torch import LazyTensor

import torch.nn.functional as F

from scipy.special import gamma
from .proposals import Proposal


numpy = lambda x: x.cpu().numpy()


def squared_distances(x, y):
    x_i = LazyTensor(x[:, None, :])  # (N,1,D)
    y_j = LazyTensor(y[None, :, :])  # (1,M,D)
    return ((x_i - y_j) ** 2).sum(dim=2)  # (N,M,1)


def normalize(points):
    return F.normalize(points, p=2, dim=1)


class EuclideanSpace(object):
    """Euclidean space R^D."""

    def __init__(self, dimension=1, dtype=None, resolution=200):
        """Creates a Euclidean space."""

        self.dimension = dimension

        if dtype is None:
            dtype = (
                torch.cuda.FloatTensor
                if torch.cuda.is_available()
                else torch.FloatTensor
            )
        self.dtype = dtype

        # Display: create a uniform grid on the unit square
        self.resolution = resolution
        ticks = np.linspace(0, 1, self.resolution + 1)[:-1] + 0.5 / self.resolution

        if self.dimension == 1:
            self.grid = torch.from_numpy(ticks).type(dtype).view(-1, 1)
        else:
            x_g, y_g = np.meshgrid(ticks, ticks)
            self.grid = (
                torch.from_numpy(np.vstack((x_g.ravel(), y_g.ravel())).T)
                .type(dtype)
                .contiguous()
            )

    def apply_noise(self, x, v):
        """Translates a noise v in the neighborhood of the identity to a position x."""

        return x + v

    def discrepancy(self, x, y):
        """Computes the energy distance between two samples."""

        n, m = len(x), len(y)

        D_xx = squared_distances(x, x).sqrt().sum(dim=1).sum()
        D_xy = squared_distances(x, y).sqrt().sum(dim=1).sum()
        D_yy = squared_distances(y, y).sqrt().sum(dim=1).sum()

        return D_xy / (n * m) - 0.5 * (D_xx / (n * n) + D_yy / (m * m))

    def scatter(self, points, color, ax=None):
        """Displays a sample as a point cloud or a log-histogram."""

        if ax is None:
            ax = plt.gca()

        if self.dimension == 1:  # display as a log-histogram
            ax.hist(
                numpy(points),
                bins=self.resolution,
                range=(0, 1),
                color=color,
                histtype="step",
                density=True,
                log=True,
            )

        else:  # display as a good old point cloud
            xy = numpy(points[:, :2])
            ax.scatter(xy[:, 0], xy[:, 1], 200 / len(xy), color=color, zorder=4)

    def plot(self, potential, color, ax=None):
        """Displays a potential on R^D as a log-density or contour plot."""

        if ax is None:
            ax = plt.gca()

        if self.dimension == 1:
            log_heatmap = numpy(potential(self.grid))
            ax.plot(numpy(self.grid).ravel(), np.exp(-log_heatmap.ravel()), color=color)

        elif self.dimension == 2:
            log_heatmap = numpy(potential(self.grid))
            log_heatmap = log_heatmap.reshape(self.resolution, self.resolution)

            scale = np.amax(np.abs(log_heatmap[:]))
            levels = np.linspace(-scale, scale, 41)

            ax.contour(
                log_heatmap,
                origin="lower",
                linewidths=1.0,
                colors=color,
                levels=levels,
                extent=(0, 1, 0, 1),
            )
        else:
            None

    def draw_frame(self, ax=None):
        if ax is None:
            ax = plt.gca()

        if self.dimension == 1:
            ax.axis([0, 1, 1e-3, 3e1])
        else:
            ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], "k")  # unit square
            ax.axis("equal")
            ax.axis([0, 1, 0, 1])
            ax.axis("off")


class Mixture(object):
    """Abstract class for mixture models."""

    def __init__(self, space, means, deviations, weights):
        """Creates a mixture model."""

        self.D = space.dimension
        self.dtype = space.dtype
        self.m = means
        self.s = deviations
        self.w = weights

    def potential(self, x):
        """Evaluates the potential on the point cloud x."""

        D_ij = squared_distances(x, self.m)  # (N,M,1)
        s_j = LazyTensor(self.s[None, :, None])  # (1,M,1)
        w_j = LazyTensor(self.w[None, :, None])  # (1,M,1)

        V_i = self.log_density(D_ij, s_j, w_j)
        return V_i.reshape(-1)  # (N,)

    def sample(self, N=1):
        """Returns a sample array of shape (N,D)."""
        y = self.sample_noise(N)
        classes = np.random.choice(len(self.w), N, p=numpy(self.w))
        y = self.m[classes] + self.s[classes].view(-1, 1) * y
        return y

    def log_density(self, D_ij, s_j, w_j):
        raise NotImplementedError()

    def sample_noise(self, N=1):
        raise NotImplementedError()


class GaussianMixture(Mixture):
    """Gaussian Mixture Model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def log_density(self, D_ij, s_j, w_j):
        D = self.D

        logK_ij = (
            -D_ij / (2 * s_j ** 2)
            + w_j.log()
            - (D / 2) * float(np.log(2 * np.pi))
            - D * s_j.log()
        )  # (N,M,1)
        V_i = -logK_ij.logsumexp(dim=1)  # (N,1), genuine torch Tensor

        return V_i.reshape(-1)  # (N,)

    def sample_noise(self, N=1):
        return torch.randn(N, self.D).type(self.dtype)


class CauchyMixture(Mixture):
    """Mixture of Cauchy distributions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.D != 1:
            raise NotImplementedError()

    def log_density(self, D_ij, s_j, w_j):
        D = self.D

        K_ij = 1 / (float(np.pi) * s_j * (1 + D_ij / (s_j ** 2)))  # (N,M,1)
        V_i = -K_ij.sum(dim=1).log()  # (N,1), genuine torch Tensor

        return V_i.reshape(-1)  # (N,)

    def sample_noise(self, N=1):
        y = torch.rand(N, self.D).type(self.dtype)
        return (np.pi * (y - 0.5)).tan()


class UnitPotential(object):
    """Arbitrary potential on the unit hypercube of dimension D."""

    def __init__(self, space, potential):
        """The minimum of the potential over [0,1]^D should be 0."""

        self.D = space.dimension
        self.dtype = space.dtype
        self.inner_potential = potential

    def potential(self, x):
        """Evaluates the potential on the point cloud x."""

        V_i = self.inner_potential(x)

        out_of_bounds = (x - 0.5).abs().max(dim=1).values > 0.5
        V_i[out_of_bounds] = 10000000.0

        return V_i.reshape(-1)  # (N,)

    def sample(self, N=1):
        """Returns a sample array of shape (N,D), computed by rejection sampling."""
        x = torch.rand(N, self.D).type(self.dtype)
        uniform = torch.rand(N).type(self.dtype)

        threshold = (-self.potential(x)).exp()
        reject = (uniform > threshold).view(-1)
        M = int(reject.sum())

        if M == 0:
            return x
        else:
            x[reject] = self.sample(M)
            return x


class BallProposal(Proposal):
    """Uniform proposal on a ball of R^D."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample_noise(self, N, scales):
        """Returns a sample of size N with ball radii given by the scales."""
        r = torch.rand(N, 1).type(self.dtype) ** (1 / self.D)  # Radius
        n = torch.randn(N, self.D).type(self.dtype)
        n = normalize(n)  # Direction, randomly sampled on the sphere

        return scales * r * n

    def nlog_density(self, target, source, log_weights, scales, probas=None):
        """Negative log-likelihood of the proposal generated by the source onto the target."""

        D_ij = squared_distances(target, source)
        neighbors_ij = (scales ** 2 - D_ij).step()  # 1 if |x_i-y_j| <= e, 0 otherwise
        volumes = float(np.pi ** (self.D / 2) / gamma(self.D / 2 + 1)) * (
            scales ** self.D
        )
        neighbors_ij = neighbors_ij / volumes

        if log_weights is None:
            neighbors_ij = neighbors_ij / len(source)
        else:
            w_j = LazyTensor(log_weights[None, :, None].exp())
            neighbors_ij = neighbors_ij * w_j

        densities_i = neighbors_ij.sum(axis=1)  # (N,K)

        if probas is None:
            logdens = - densities_i.log()
            logdens[densities_i < 1e-5] = 10000
            return logdens
        else:
            return -(densities_i * probas[None, :]).sum(dim=1).log().view(-1)


def local_moments(points, radius=1, ranges=None):
    # print(radius)

    # B, N, D = points.shape
    shape_head, D = points.shape[:-1], points.shape[-1]

    scale = 1.41421356237 * radius  # math.sqrt(2) is not super JIT-friendly...
    x = points / scale  # Normalize the kernel size

    # Computation:    
    x = torch.cat((torch.ones_like(x[...,:1]), x), dim = -1)  # (B, N, D+1)

    x_i = LazyTensor(x[...,:,None,:])  # (B, N, 1, D+1)
    x_j = LazyTensor(x[...,None,:,:])  # (B, 1, N, D+1)

    D_ij = ((x_i - x_j) ** 2).sum(-1)  # (B, N, N), squared distances
    K_ij = (- D_ij).exp()  # (B, N, N), Gaussian kernel

    C_ij = (K_ij * x_j).tensorprod(x_j)  # (B, N, N, (D+1)*(D+1))

    C_ij.ranges = ranges
    C_i  = C_ij.sum(dim = len(shape_head)).view(shape_head + (D+1, D+1))  # (B, N, D+1, D+1) : descriptors of order 0, 1 and 2

    w_i = C_i[...,:1,:1]               # (B, N, 1, 1), weights
    m_i = C_i[...,:1,1:] * scale       # (B, N, 1, D), sum
    c_i = C_i[...,1:,1:] * (scale**2)  # (B, N, D, D), outer products

    mass_i = w_i.squeeze(-1).squeeze(-1)  # (B, N)
    dev_i = (m_i / w_i).squeeze(-2) - points  # (B, N, D)
    cov_i  = (c_i - (m_i.transpose(-1, -2) * m_i) / w_i) / w_i  # (B, N, D, D)

    return mass_i, dev_i, cov_i


def svd(cov):
    # https://github.com/pytorch/pytorch/issues/41306
    usv = torch.svd(cov.cpu())
    return usv.U.to(cov.device), usv.S.to(cov.device), usv.V.to(cov.device)


class GaussianProposal(Proposal):
    """Gaussian proposal in R^D."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptive = False

    def adapt(self, x):
        self.adaptive = True
        mass_i, dev_i, cov_i = local_moments(x, radius = self.s[0])
        self.means = x + dev_i  # (N, D)
        self.covariances = cov_i  # (N, D, D)
        U, S, V = svd(self.covariances)
        self.covariances_half = U @ (S.sqrt()[...,:,None] * V.transpose(1,2))
        self.covariances_inv  = U @ ((1 / S)[...,:,None] * V.transpose(1,2))
        self.log_det_cov_half = S.log().sum(-1) / 2
        # print(S.sqrt().mean())

    def adaptive_sample(self, x, indices):
        noise = self.sample_noise(len(x), 1)  # (N, D)
        means = self.means[indices]
        cov_half = self.covariances_half[indices]
        # print(means.shape, cov_half.shape, noise.shape)
        return means + (cov_half @ noise[:,:,None]).squeeze(-1)


    def sample_noise(self, N, scales):
        """Returns a sample of size N with ball radii given by the scales."""

        return scales * torch.randn(N, self.D).type(self.dtype)

    def nlog_density(self, target, source, log_weights, scales, probas=None):
        """Negative log-likelihood of the proposal generated by the source onto the target."""

        if self.adaptive:

            x_i = LazyTensor(target[:, None, :])  # (N,1,D)
            y_j = LazyTensor(source[None, :, :])  # (1,M,D)
            s_j = self.covariances_inv  # (M, D, D)
            s_j = LazyTensor(s_j.view(s_j.shape[0], -1)[None, :, :])  # (1, M, D*D)
            D_ij = (x_i - y_j) | s_j.matvecmult(x_i - y_j)  # (N,M,1)

            det_j = LazyTensor(self.log_det_cov_half[None, :, None])

            logK_ij = (
                -D_ij / 2
                - (self.D / 2) * float(np.log(2 * np.pi))
                - det_j
            )

        else:
            D_ij = squared_distances(target, source)
            logK_ij = (
                -D_ij / (2 * scales ** 2)
                - (self.D / 2) * float(np.log(2 * np.pi))
                - self.D * scales.log()
            )

        if log_weights is None:
            logK_ij = logK_ij - float(np.log(len(source)))
        else:
            logW_j = LazyTensor(log_weights[None, :, None])
            logK_ij = logK_ij + logW_j

        logdensities_i = logK_ij.logsumexp(dim=1).reshape(-1)  # (N,K)

        if probas is None:
            return -logdensities_i
        else:
            return -(logdensities_i + probas.log()[None, :]).logsumexp(dim=1).view(-1)





class GMMProposal(Proposal):
    """Gaussian Mixture Model proposal in R^D."""

    def __init__(self, *args, n_classes = 5, covariance_type='full', n_iter = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.K = n_classes
        self.covariance_type = covariance_type
        self.n_iter = n_iter

        self.means = .5 + .5 * torch.randn(self.K, self.D).type(self.dtype)
        self.weights = torch.rand(self.K).type(self.dtype)
        self.weights /= self.weights.sum()

        if covariance_type == "full":
            self.covariances = torch.empty(self.K, self.D, self.D).type(self.dtype)
            for k in range(self.K):
                self.covariances[k, :, :] = torch.diag(1 + 0 * torch.rand(self.D).type(self.dtype)) ** 2
            
        else:
            self.covariances = torch.empty(self.K, self.D, 1).type(self.dtype)
            for k in range(self.K):
                self.covariances[k, :, :] = (1 + 0 * torch.rand(self.D, 1).type(self.dtype)) ** 2


    def adapt(self, x):
        N = len(x)
        K, D = self.K, self.D

        # EM iterations
        for _ in range(self.n_iter):

            x_i = LazyTensor(x.view(N, 1, D))
            m_j = LazyTensor(self.means.view(1, K, D))

            if  self.covariance_type == 'full':
                precision = self.covariances.inverse()
                prec_j = LazyTensor(precision.reshape(1, K, D * D))
                D_ij = ((x_i - m_j) | (prec_j).matvecmult(x_i - m_j))
                w_j = LazyTensor(self.weights.view(1, K, 1) * torch.sqrt(precision.det()).view(1, K, 1))
            else:
                cov_j = LazyTensor(self.covariances.view(1, K, D) + .0000001)
                D_ij = ((x_i - m_j) ** 2 / cov_j).sum(dim=2)
                w_j = LazyTensor(self.weights.view(1, K, 1) * torch.rsqrt(torch.prod(self.covariances, dim=1) + .0000001).view(1, K, 1))

            Gauss_ij = (- D_ij / 2).exp() * w_j
            BayesNorm_i = LazyTensor((Gauss_ij.sum(dim=1) + .0000001).view(N, 1, 1))

            # membership probabilities H: a LazyTensor of size N, K
            H_ij = (Gauss_ij / BayesNorm_i) # N x K

            H_sum = H_ij.sum(dim=0) # 1 x K
            self.weights = H_sum.view(-1) / N
            self.weights /= self.weights.sum()

            self.means = (H_ij * x_i).sum(dim=0) / (H_sum + .0000001)

            m_j = LazyTensor(self.means.view(1, K, D))
            if self.covariance_type == 'full':
                self.covariances = (H_ij * (x_i - m_j).tensorprod(x_i - m_j)).sum(0).view(K, D, D) / (H_sum.view(K, 1, 1) + .0000001)
            else:
                self.covariances = (H_ij * (x_i - m_j) ** 2).sum(0) / (H_sum.view(K,1) + .0000001)

            assert not torch.isnan(self.means).sum()
            assert not torch.isnan(self.covariances).sum()
            assert not torch.isnan(self.weights).sum()

        #labels = H_ij.argmax(dim=1)
        #assert not torch.isnan(labels).sum()

        if self.covariance_type == "full":
            for k in range(self.K):
                self.covariances[k, :, :] += torch.diag(.01 + 0 * torch.rand(self.D).type(self.dtype)) ** 2
        else:
            for k in range(self.K):
                self.covariances[k, :, :] += (.01 + 0 * torch.rand(self.D, 1).type(self.dtype)) ** 2


        U, S, V = svd(self.covariances)
        self.covariances_half = U @ (S.sqrt()[...,:,None] * V.transpose(1,2))
        self.covariances_inv  = U @ ((1 / S)[...,:,None] * V.transpose(1,2))
        self.log_det_cov_half = S.log().sum(-1) / 2
        # print(S.sqrt().mean())


    def adaptive_sample(self, x, indices):
        noise = torch.randn(len(x), self.D).type(self.dtype)
        classes = np.random.choice(len(self.weights), len(x), p=numpy(self.weights))

        cov_half = self.covariances_half[classes]
        y = self.means[classes] + (cov_half @ noise[:,:,None]).squeeze(-1)
        return y


    def sample_noise(self, N, scales):
        """Returns a sample of size N with ball radii given by the scales."""

        return scales * torch.randn(N, self.D).type(self.dtype)

    def nlog_density(self, target, source, log_weights, scales, probas=None):
        """Negative log-likelihood of the proposal generated by the source onto the target."""

        x_i = LazyTensor(target[:, None, :])  # (N,1,D)
        y_j = LazyTensor(self.means[None, :, :])  # (1,M,D)
        s_j = self.covariances_inv  # (M, D, D)
        s_j = LazyTensor(s_j.view(s_j.shape[0], -1)[None, :, :])  # (1, M, D*D)
        D_ij = (x_i - y_j) | s_j.matvecmult(x_i - y_j)  # (N,M,1)

        det_j = LazyTensor(self.log_det_cov_half[None, :, None])

        logK_ij = (
            -D_ij / 2
            - (self.D / 2) * float(np.log(2 * np.pi))
            - det_j
        )

        log_weights = self.weights.log()
        logW_j = LazyTensor(log_weights[None, :, None])
        logK_ij = logK_ij + logW_j

        logdensities_i = logK_ij.logsumexp(dim=1).reshape(-1)  # (N,)
        return - logdensities_i
        #return -(logdensities_i + probas.log()[None, :]).logsumexp(dim=1).view(-1)

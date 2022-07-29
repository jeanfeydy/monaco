import numpy as np
import torch
from pykeops.torch import LazyTensor

numpy = lambda x: x.cpu().numpy()


class Proposal:
    """Abstract class for proposals, handling the logic for sums of kernels."""

    def __init__(
        self, space, scale=1.0, probas=None, exploration=None, exploration_proposal=None
    ):
        """Creates a proposal, possibly with multiple scales."""

        self.space = space
        self.D = space.dimension
        self.dtype = space.dtype
        self.exploration = exploration  # None or float, < 1
        self.exploration_proposal = exploration_proposal  # Proposal

        scale = [scale] if type(scale) is float else scale

        self.s = scale
        self.probas = (
            torch.FloatTensor([1.0] * len(self.s)).type(self.dtype)
            if probas is None
            else probas
        )
        self.probas = self.probas / self.probas.sum()

        if len(self.s) != len(self.probas):
            raise ValueError("Scale and probas should have the same size.")

    def explore(self, x, y):
        """Replace a set proportion of the points by "random" samples."""
        if self.exploration is not None:
            to_explore = torch.rand(len(y)).type_as(y) <= self.exploration
            y[to_explore] = self.exploration_proposal.sample(x[to_explore])
        return y

    def sample_indices(self, x):
        """Returns samples with scale indices."""

        # Multi-kernel choice
        self.probas = self.probas / self.probas.sum()
        indices = np.random.choice(len(self.s), len(x), p=numpy(self.probas))
        indices = torch.from_numpy(indices).to(x.device).long()

        s = torch.FloatTensor(self.s).type_as(x)
        scales = s[indices].view(-1, 1)

        # Actual proposal
        y = self.space.apply_noise(x, self.sample_noise(len(x), scales))

        return self.explore(x, y), indices

    def sample(self, x):
        """Returns a random sample generated by the proposal."""

        y, _ = self.sample_indices(x)
        return y

    def potential(self, source, log_weights=None):
        """Returns a potential that may be evaluated at any location.

        Here, the source and log_weights specify a source distribution
        that is perturbated by the proposal's noise generator.
        """
        s = torch.FloatTensor(self.s).type_as(source)
        scales = LazyTensor(s)  # (1,1,K)

        V = lambda target: self.nlog_density(
            target, source, log_weights, scales, self.probas
        )

        if self.exploration is not None:
            Vexp = self.exploration_proposal.potential(source, log_weights)

            def V_total(t):
                nV_t = np.log(1 - self.exploration) - V(t)
                nVexp_t = np.log(self.exploration) - Vexp(t)

                return -torch.stack((nV_t.view(-1), nVexp_t.view(-1))).logsumexp(0)

            return V_total

        return V

    def nlog_densities(self, source, log_weights=None):
        s = torch.FloatTensor(self.s).type_as(source)
        scales = LazyTensor(s)  # (1,1,K)

        V_source = self.nlog_density(source, source, log_weights, scales)

        return V_source

    def sample_noise(self, N, scales):
        raise NotImplementedError()

    def nlog_density(self, target, source, log_weights, scales, probas=None):
        raise NotImplementedError()

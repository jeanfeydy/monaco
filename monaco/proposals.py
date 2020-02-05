import numpy as np
import torch
from pykeops.torch import LazyTensor

numpy = lambda x : x.cpu().numpy()

class Proposal():
    def __init__(self, space, scale = 1., probas = None):
        self.space = space
        self.D = space.dimension
        self.dtype = space.dtype
        
        scale = [scale] if type(scale) is float else scale

        self.s = scale
        self.probas = torch.FloatTensor([1.] * len(self.s)).type(self.dtype) if probas is None else probas
        self.probas = self.probas / self.probas.sum()

        if len(self.s) != len(self.probas):
            raise ValueError("Scale and probas should have the same size.")
    
    
    def sample_indices(self, x):
        self.probas = self.probas / self.probas.sum()
        indices = np.random.choice(len(self.s), len(x), p = numpy(self.probas))
        indices = torch.from_numpy(indices).to(x.device).long()
        
        s = torch.FloatTensor(self.s).type_as(x)
        scales = s[indices].view(-1,1)

        y = self.space.apply_noise(x, self.sample_noise(len(x), scales))
        return y, indices


    def sample(self, x):
        y, _ = self.sample_indices(x)
        return y


    def potential(self, source, log_weights = None):

        s = torch.FloatTensor(self.s).type_as(source)
        scales = LazyTensor(s)  # (1,1,K)

        V = lambda target : self.nlog_density(target, source, log_weights, scales, self.probas)

        return V
    
    def sample_noise(self, N, scales):
        raise NotImplementedError()

    def nlog_density(self, target, source, log_weights, scales, probas):
        raise NotImplementedError()




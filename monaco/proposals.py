import numpy as np
import torch
from pykeops.torch import LazyTensor

numpy = lambda x : x.cpu().numpy()

class Proposal():
    def __init__(self, space, scale = 1., logits = None):
        self.space = space
        self.D = space.dimension
        self.dtype = space.dtype
        
        scale = [scale] if type(scale) is float else scale

        self.s = scale
        self.logits = torch.FloatTensor([0.] * len(self.s)).type(self.dtype) if logits is None else logits

        if len(self.s) != len(self.logits):
            raise ValueError("Scale and logits should have the same size.")
    
    
    def sample(self, x):
        probas = self.logits.exp()
        probas = probas / probas.sum()
        scales = np.random.choice(self.s, len(x), p = numpy(probas))
        scales = torch.from_numpy(scales).type_as(x).view(-1,1)

        return self.space.apply_noise(x, self.sample_noise(len(x), scales))


    def potential(self, source, log_weights = None):

        s = torch.FloatTensor(self.s).type_as(source)
        scales = LazyTensor(s)  # (1,1,K)

        V = lambda target : self.nlog_density(target, source, log_weights, scales, self.logits)

        return V
    
    def sample_noise(self, N, scales):
        raise NotImplementedError()

    def nlog_density(self, target, source, log_weights, scales, logits):
        raise NotImplementedError()




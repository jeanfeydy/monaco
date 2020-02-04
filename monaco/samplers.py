
import numpy as np
import itertools
import torch

import seaborn as sns
from matplotlib import pyplot as plt

numpy = lambda x : x.cpu().numpy()


def display(space, potential, sample, proposal_sample=None, proposal_potential=None, true_sample=None):

    if true_sample is not None:
        space.scatter(true_sample, "red")

    if proposal_sample is not None:
        space.scatter(proposal_sample, "green")
    if proposal_potential is not None:
        space.plot(proposal_potential, "green")

    space.plot(potential, "red")
    space.scatter(sample, "blue")

    space.draw_frame()


def display_samples(sampler, iterations = 100, runs = 5):

    verbosity = sampler.verbose
    sampler.verbose = True
    x_prev = sampler.x
    
    iters, rates, errors, fluctuations = [], [], [], []

    for run in range(runs):

        to_plot = [1, 2, 5, 10, 50, 100]
        

        for it, (x, y, rate, u) in enumerate(sampler):
            
            iters.append(it)
            rates.append(rate)

            try:
                N = len(x)
                errors.append( sampler.space.discrepancy(x, sampler.distribution.sample(N)) )
                fluctuations.append( sampler.space.discrepancy(sampler.distribution.sample(N), sampler.distribution.sample(N)))
            except AttributeError:
                None

            if run == 0 and it + 1 in to_plot:
                plt.figure(figsize = (12,8))

                try:
                    display(sampler.space, sampler.distribution.potential, x, y, 
                            sampler.proposal.potential(x_prev, u), 
                            sampler.distribution.sample(len(x)) )
                except AttributeError:
                    display(sampler.space, sampler.distribution.potential, x, y)

                plt.title(f"it = {it+1}")
                plt.tight_layout()

            x_prev = x

            if it > iterations:
                break

    iters, rates = np.array(iters), np.array(rates)

    plt.figure(figsize=(12,8))
    sns.lineplot(x = np.array(iters), y = np.array(rates), markers = "*", label="Acceptance rate")
    plt.ylim(0,1)
    plt.xlabel("Iterations")
    plt.tight_layout()

    if errors != []:
        plt.figure(figsize=(12,8))
        errors = np.array(errors)
        sns.lineplot(x = iters, y = errors, markers = "*", label="Error")

    if fluctuations != []:
        fluctuations = np.array(fluctuations)
        sns.lineplot(x = iters, y = fluctuations, markers = ".", label="Fluctuations")
        plt.xlabel("Iterations")
        plt.tight_layout()

    sampler.verbose = verbosity





class MonteCarloSampler(object):

    def __init__(self, space, start, proposal, verbose = False):
        self.space = space
        self.x = start
        self.proposal = proposal
        self.verbose = verbose
        self.iteration = 0

    def fit(self, distribution):
        self.distribution = distribution
        return self

    def __iter__(self):
        return self

    def __next__(self):
        x, y, accept, u = self.update()
        self.iteration += 1

        if self.verbose:
            return x, y, accept, u
        else:
            return x

    def update(self):
        raise NotImplementedError()


class CMC(MonteCarloSampler):
    """Collective Monte-Carlo."""

    def __init__(self, space, start, proposal, annealing = None, verbose = False):
        super().__init__(space, start, proposal, verbose = verbose)
        self.annealing = annealing

    def update(self):
        x = self.x
        N = len(x)
        indices = torch.randint(N, size = (N,)).to(x.device)
        y = self.proposal.sample(x[indices,:])  # Proposal

        # Annealing ratio
        ratio = 1 if self.annealing is None else 1 - np.exp(- self.iteration / self.annealing)

        # Logarithm of the CMC ratio:
        scores = ratio * (self.distribution.potential(x) - self.distribution.potential(y)) \
                + self.proposal.potential(x)(y) - self.proposal.potential(x)(x)

        accept = torch.rand(N).type_as(x) <= scores.exp()  # h(u) = min(1, u)

        x[accept,:] = y[accept,:]  # MCMC update
        
        # x = x.clamp(0, 1)       # Clip to the unit square
        rate = (1. * accept).mean()

        self.x = x
        return x, y, rate, None




class KIDS_CMC(MonteCarloSampler):
    """Kernel Importance-by-Deconvolution Sampling Collective Monte-Carlo."""

    def __init__(self, space, start, proposal, annealing = None, verbose = False, iterations = 100):
        super().__init__(space, start, proposal, verbose = verbose)
        self.annealing = annealing
        self.nits = iterations

    def update(self):
        x = self.x
        N = len(x)
        indices = torch.randint(N, size = (N,)).to(x.device)
        y = self.proposal.sample(x[indices,:])  # Proposal

        # Annealing ratio
        ratio = 1 if self.annealing is None else 1 - np.exp(- self.iteration / self.annealing)
        V_x = self.distribution.potential(x)


        # Richardson-Lucy-like iterations ----------------------------------
        # We look for u such that
        #   proposal.potential(x, u)(x_i) = - log( k * e^u ) (x_i) = q * V(x_i)
        #
        target = - ratio * V_x
        target = target - target.logsumexp(0)  # Normalize the target log-likelihood

        #u = (- np.log(N) * np.ones(N)).astype(dtype)
        u = target
        for it in range(self.nits):
            offset = target + self.proposal.potential(x, u)(x)
            # offset = - proposal.potential(x, offset)(x)  # Genuine Richardson-Lucy would have this line too
            u = u + offset

        u = u - u.logsumexp(0)  # Normalize the proposal


        # Importance sampling: ---------------------------------------------
        indices = np.random.choice(N, size = N, p = numpy(u.exp()))
        indices = torch.from_numpy(indices).to(x.device)
        y = self.proposal.sample(x[indices,:])  # Proposal

        # Logarithm of the CMC ratio:
        scores = ratio * (V_x - self.distribution.potential(y)) \
                + self.proposal.potential(x, u)(y) - self.proposal.potential(x, u)(x)

        accept = torch.rand(N).type_as(x) <= scores.exp()  # h(u) = min(1, u)

        x[accept,:] = y[accept,:]  # MCMC update
        rate = (1. * accept).mean()

        self.x = x
        return x, y, rate, u

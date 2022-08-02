from cmath import inf
from unicodedata import numeric
import numpy as np
import itertools
import torch

import seaborn as sns
from matplotlib import pyplot as plt

from scipy.special import gamma

numpy = lambda x: x.cpu().numpy()

FIGSIZE_LARGE = (4, 4)
FIGSIZE = (8, 12)  # Small thumbnails for the paper
CELLSIZE = (4, 4)
FIGSIZE_INFO = (8, 8)  # Small thumbnails for the paper


def display(
    space,
    potential,
    sample,
    proposal_sample=None,
    proposal_potential=None,
    true_sample=None,
):
    """Fancy display of the current state of a Monte Carlo sampler."""

    if proposal_sample is not None:
        space.scatter(proposal_sample, "green")

    space.plot(potential, "red")
    space.scatter(sample, "blue")

    space.draw_frame()


def display_samples(sampler, iterations=100, to_plot=None, runs=5, small=True):
    """Displays results and statistics for a run of a Monte Carlo sampler."""

    if to_plot is None:
        to_plot = [1, 2, 5, 10, 20, 50, 80, 100]

    to_plot = [it for it in to_plot if it <= iterations]

    verbosity = sampler.verbose
    sampler.verbose = True

    start = sampler.x.clone()

    iters, rates, errors, fluctuations, probas, constants, number_of_neighbours, ESS = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    # Initial error
    try:
        N = len(start)
        # "Energy distance" between a MCMC sample and a genuine one
        errors.append(
            sampler.space.discrepancy(start, sampler.distribution.sample(N)).item()
        )
        # "Energy distance" between two genuine samples
        fluctuations.append(
            sampler.space.discrepancy(
                sampler.distribution.sample(N), sampler.distribution.sample(N)
            ).item()
        )
    except AttributeError:
        None

    # We run the sampler several times to aggregate statistics
    # and display fancy viualizations for the last run:
    for run in range(runs):
        x_prev = start.clone()  # We copy the initialization to make independent runs!
        sampler.x[:] = start.clone()  # in-place update of the sampler state
        sampler.iteration = 0

        if run == runs - 1:  # Fancy display for the last run

            if small:
                nrows = int((len(to_plot) + 1) / 2) + ((len(to_plot) + 1) % 2 > 0)
                plt.figure(figsize=(CELLSIZE[0] * 2, CELLSIZE[1] * nrows))
                plt.subplot(nrows, 2, 1)
                fig_index = 2
            else:
                plt.figure(figsize=FIGSIZE_LARGE)

            display(sampler.space, sampler.distribution.potential, x_prev)

            plt.title(f"it = 0")
            plt.tight_layout()

        # N.B.: our samplers are implemented as Python iterators.
        for it, info in enumerate(sampler):

            # Unwrap the full sampling "information":
            x = info["sample"]  # The actual points
            y = info.get(
                "proposal", None
            )  # samples that have been accepted or rejected
            u = info.get("log-weights", None)  # Deconvolution log-weights

            iters.append(it + 1)

            # Save the relevant monitoring information:
            try:  # Acceptance rate
                rates.append(info["rate"].item())
            except KeyError:
                None

            try:  # Kernel probabilities for MOKA
                probas.append(info["probas"])
            except KeyError:
                None

            try:  # Estimation of the "total mass" of the distribution
                constants.append(info["normalizing constant"].item())
            except KeyError:
                None

            try:  # Count the number of neighbours
                number_of_neighbours.append(info["number of neighbours"])
            except KeyError:
                None

            try:  # Estimation of the ESS
                ESS.append(info["ESS"].item())
            except KeyError:
                None

            try:
                N = len(x)
                # "Energy distance" between a MCMC sample and a genuine one
                errors.append(
                    sampler.space.discrepancy(x, sampler.distribution.sample(N)).item()
                )
                # "Energy distance" between two genuine samples
                fluctuations.append(
                    sampler.space.discrepancy(
                        sampler.distribution.sample(N), sampler.distribution.sample(N)
                    ).item()
                )
            except AttributeError:
                None

            # On the last run, display some fancy visualizations:
            if run == runs - 1 and it + 1 in to_plot:

                if small:
                    plt.subplot(nrows, 2, fig_index)
                    fig_index += 1
                else:
                    plt.figure(figsize=FIGSIZE_LARGE)

                try:
                    display(
                        sampler.space,
                        sampler.distribution.potential,
                        x,
                        y,
                        sampler.proposal.potential(x_prev, u),
                        sampler.distribution.sample(len(x)),
                    )
                except AttributeError:
                    display(sampler.space, sampler.distribution.potential, x, y)

                plt.title(f"it = {it+1}")
                plt.tight_layout()

            x_prev = x

            if it > iterations:
                break

    iters = np.array(iters)

    nfigures = 0
    for dat in [rates, errors, probas, constants, number_of_neighbours, ESS]:
        if dat != []:
            nfigures += 1

    nrows_data = (nfigures + 1) // 2
    if small:
        plt.figure(figsize=(CELLSIZE[0] * 2, CELLSIZE[1] * nrows_data))
        fig_index = 1

    # Overview for the acceptance rates:
    if rates != []:
        rates = np.array(rates)

        if small:
            plt.subplot(nrows_data, 2, fig_index)
            fig_index += 1
        else:
            plt.figure(figsize=FIGSIZE_LARGE)
        sns.lineplot(
            x=np.array(iters),
            y=np.array(rates),
            marker="o",
            markersize=6,
            label="Acceptance rate",
            ci="sd",
        )
        plt.ylim(0, 1)
        plt.xlabel("Iterations")
        plt.gca().set_box_aspect(1)
        if small:
            plt.tight_layout()

    # Overview for the Energy Distances between MCMC and genuine samples:
    if errors != []:
        errors = np.array(errors)

        if small:
            plt.subplot(nrows_data, 2, fig_index)
            fig_index += 1
        else:
            plt.figure(figsize=FIGSIZE_LARGE)

        sns.lineplot(
            x=np.insert(iters, 0, 0),
            y=errors,
            marker="o",
            markersize=6,
            label="Error",
            ci="sd",
        )

    # Overview for the Energy Distances between two genuine samples:
    if fluctuations != []:
        fluctuations = np.array(fluctuations)

        sns.lineplot(
            x=np.insert(iters, 0, 0),
            y=fluctuations,
            marker="X",
            markersize=6,
            label="Fluctuations",
            ci="sd",
        )
        plt.xlabel("Iterations")
        plt.ylim(bottom=0.0)
        plt.gca().set_box_aspect(1)
        if small:
            plt.tight_layout()

    # Overview for the MOKA kernel weights:
    if probas != []:
        probas = numpy(torch.stack(probas)).T

        if small:
            plt.subplot(nrows_data, 2, fig_index)
            fig_index += 1
        else:
            plt.figure(figsize=FIGSIZE_LARGE)

        markers = itertools.cycle(("o", "X", "P", "D", "^", "<", "v", ">", "*"))
        for scale, proba, marker in zip(sampler.proposal.s, probas, markers):
            sns.lineplot(
                x=iters,
                y=proba,
                marker=marker,
                markersize=6,
                label="scale = {:.3f}".format(scale),
                ci="sd",
            )
        plt.xlabel("Iterations")
        plt.ylim(bottom=0.0)
        plt.gca().set_box_aspect(1)
        if small:
            plt.tight_layout()

    # Overview for the normalizing constants:
    if constants != []:

        if small:
            plt.subplot(nrows_data, 2, fig_index)
            fig_index += 1
        else:
            plt.figure(figsize=FIGSIZE_LARGE)

        constants = np.array(constants)
        sns.lineplot(
            x=iters,
            y=constants,
            marker="o",
            markersize=6,
            label="Normalizing constant",
            ci="sd",
        )

        plt.xlabel("Iterations")
        plt.ylim(bottom=0.0)
        plt.gca().set_box_aspect(1)
        if small:
            plt.tight_layout()

    # Overview for the number of neighbours:
    if number_of_neighbours != []:
        number_of_neighbours = numpy(torch.stack(number_of_neighbours)).T

        if small:
            plt.subplot(nrows_data, 2, fig_index)
            fig_index += 1
        else:
            plt.figure(figsize=FIGSIZE_LARGE)

        markers = itertools.cycle(("o", "X", "P", "D", "^", "<", "v", ">", "*"))
        for scale, nb_neigh, marker in zip(
            sampler.proposal.s, number_of_neighbours, markers
        ):
            sns.lineplot(
                x=iters,
                y=nb_neigh,
                marker=marker,
                markersize=6,
                label="scale = {:.3f}".format(scale),
                ci="sd",
            )
        plt.xlabel("Iterations")
        plt.ylabel("Mean number of neighbors")
        plt.ylim(bottom=0.0)
        plt.gca().set_box_aspect(1)
        if small:
            plt.tight_layout()

    # Overview for the ESS:
    if ESS != []:

        try:
            ESSmax = sampler.ESSmax
        except AttributeError:
            ESSmax = 0.0

        if small:
            plt.subplot(nrows_data, 2, fig_index)
            fig_index += 1
        else:
            plt.figure(figsize=FIGSIZE_LARGE)

        ESS = np.array(ESS)
        sns.lineplot(
            x=iters,
            y=ESS,
            marker="o",
            markersize=6,
            label="ESS",
            ci="sd",
        )
        plt.hlines(ESSmax, 0, iterations, "k", "dotted")

        plt.xlabel("Iterations")
        plt.ylim(bottom=0.0)
        plt.gca().set_box_aspect(1)
        if small:
            plt.tight_layout()

    sampler.verbose = verbosity

    to_return = {
        "iteration": np.insert(iters, 0, 0),
        "rate": rates,
        "normalizing constant": constants,
        "error": errors,
        "fluctuation": fluctuations,
        "probas": probas,
        "number of neighbours": number_of_neighbours,
        "ESS": ESS,
    }

    return to_return


# Sampling classes =====================================================


class MonteCarloSampler(object):
    """Abstract Monte Carlo sampler, as a Python iterator."""

    def __init__(self, space, start, proposal, verbose=False):
        self.space = space
        self.x = start.clone()
        self.proposal = proposal
        self.verbose = verbose
        self.iteration = 0

    def fit(self, distribution):
        self.distribution = distribution
        return self

    # N.B.: Our samplers are defined as Python iterators.
    def __iter__(self):
        return self

    def __next__(self):
        info = self.update()
        self.iteration += 1

        if self.verbose:
            return info
        else:
            return info["sample"]

    def update(self):
        raise NotImplementedError()


# Baselines ===========================================


class ParallelMetropolisHastings(MonteCarloSampler):
    """Parallel Metropolis-Hastings algorithm."""

    def __init__(self, space, start, proposal, annealing=None, verbose=False):
        super().__init__(space, start, proposal, verbose=verbose)
        self.annealing = annealing

    def update(self):
        x = self.x
        N = len(x)
        y = self.proposal.sample(x)  # Proposal

        # Annealing ratio
        ratio = (
            1
            if self.annealing is None
            else 1 - np.exp(-self.iteration / self.annealing)
        )

        # Logarithm of the MH ratio:
        scores = ratio * (
            self.distribution.potential(x) - self.distribution.potential(y)
        )

        accept = torch.rand(N).type_as(x) <= scores.exp()  # h(u) = min(1, u)

        x[accept, :] = y[accept, :]  # MCMC update

        # x = x.clamp(0, 1)       # Clip to the unit square
        rate = (1.0 * accept).mean()

        self.x = x

        info = {
            "sample": x,
            "proposal": y,
            "rate": rate,
        }

        return info


class SAIS(MonteCarloSampler):
    """Safe adaptive importance sampling, by batch for the sake of efficiency on the GPU."""

    def __init__(
        self,
        space,
        start,
        proposal,
        annealing=False,
        scale0=1.0,
        q0=None,
        N=None,
        sample_size=None,
        T0=1,
        verbose=False,
    ):
        super().__init__(space, start, proposal, verbose=verbose)
        self.annealing = annealing
        self.q0 = q0
        self.N = N  # Batch size
        self.N0 = int(N * T0 / 2.0) if annealing else N
        self.sample_size = N if sample_size is None else sample_size
        self.T0 = T0  # Burn in phase
        self.scale0 = scale0  # Initial size of the proposal
        self.dtype = space.dtype

    def importance_sampling(self, n):
        log_weights = self.scores - self.scores.logsumexp(0)
        indices = np.random.choice(
            len(self.memory),
            size=n,
            p=numpy(log_weights.exp() / log_weights.exp().sum()),
        )
        indices = torch.from_numpy(indices).to(self.memory.device)
        return self.memory[indices, :]

    def update(self):

        if self.iteration == 0:
            # Add N0 defensers initially
            self.memory = self.q0.sample(self.N)
            self.scores = self.q0.potential(self.memory) - self.distribution.potential(
                self.memory
            )  # Pi / Q0
            if self.annealing:
                self.scores *= 0.75

        if self.annealing:
            annealing_factor = (1 + self.N * self.iteration / self.N0) ** (
                -1.0 / (4 + self.space.dimension)
            )
            if self.iteration < self.T0:
                # Annealing ratio: weight of the defensive sample
                # lambda_t = (
                #     0.0 if self.annealing is None else np.exp(-self.iteration / self.annealing)
                # )
                lambda_t = 1.0 if self.iteration < self.T0 / 2.0 else 0.5
                score_smoothing = 0.75
            else:
                # Then use the annealing factor
                lambda_t = 0.25 * annealing_factor
                score_smoothing = 1.0

            self.proposal.s = [self.scale0 * annealing_factor]
        else:
            score_smoothing = 1.0
            lambda_t = 0.0
            self.proposal.s = [self.scale0]

        # Choose to sample from the mixture or the defensive initialization:
        # defensive == 0 if mixture, 1 if sample from q0:
        defensive = torch.rand(self.N).type(self.dtype) < lambda_t

        n_defense = int((1.0 * defensive).sum().item())
        defenders = self.q0.sample(n_defense)

        attackers = self.importance_sampling(self.N - n_defense)
        attackers = self.proposal.sample(attackers)

        new_points = torch.cat((defenders, attackers), dim=0)
        # Potential for the new points:
        new_potentials = self.distribution.potential(new_points)

        # Compute the scores for the attackers and defenders:
        defense_scores = self.q0.potential(new_points)

        log_weights = self.scores - self.scores.logsumexp(0)
        mixture_scores = self.proposal.potential(self.memory, log_weights)(new_points)

        if lambda_t == 1.0:
            new_scores = defense_scores
        elif lambda_t == 0.0:
            new_scores = mixture_scores
        else:
            new_scores = -torch.logsumexp(
                torch.stack(
                    (
                        np.log(1 - lambda_t) - mixture_scores,
                        np.log(lambda_t) - defense_scores,
                    )
                ),
                0,
            )

        new_scores -= new_potentials
        new_scores *= score_smoothing

        #         new_scores = (
        #             -(
        #                 (1 - lambda_t) * (-mixture_scores).exp()
        #                 + lambda_t * (-defense_scores).exp()
        #             ).log()
        #             - new_potentials
        #         ) * score_smoothing

        # Add to memory and scores:
        self.memory = torch.cat((self.memory, new_points), dim=0)
        self.scores = torch.cat((self.scores, new_scores), dim=0)

        # Return a sample of size sample_size:
        x = self.importance_sampling(min(self.sample_size, len(self.memory)))

        info = {
            "sample": x,
        }

        return info


class SMC(MonteCarloSampler):
    """Sequential Monte Carlo"""

    def __init__(
        self, space, start, V0, proposal, temp, ESSmax, mh_step=1, verbose=False
    ):
        super().__init__(space, start, proposal, verbose=verbose)
        self.V = V0
        self.V0 = V0
        self.N = len(self.x)
        self.weights = 1.0 / self.N * torch.ones(self.N, device=self.x.device)
        self.temp = temp
        self.ESSmax = ESSmax
        self.mh_step = mh_step

    def ESS(self):
        return 1.0 / (self.weights**2).sum()

    def update(self):

        iter = self.iteration
        a = min((1 + iter) / self.temp, 1.0)
        Vtemp = lambda x: (1 - a) * self.V0(x) + a * self.distribution.potential(x)

        # Update weights (note: does not depend on the next sample)
        log_weights = self.weights.log() - Vtemp(self.x) + self.V(self.x)
        log_weights -= log_weights.logsumexp(0)
        self.weights = log_weights.exp()

        # Resampling
        ESS = self.ESS()
        if ESS < self.ESSmax:
            index = self.weights.multinomial(num_samples=self.N, replacement=True)
            self.x = self.x[index, :]
            self.weights = 1.0 / self.N * torch.ones(self.N, device=self.x.device)

        x = self.x

        # Metropolis-Hastings with target Vtemp
        for k in range(self.mh_step):
            y = self.proposal.sample(x)  # Proposal

            # Logarithm of the MH ratio:
            scores = Vtemp(x) - Vtemp(y)

            accept = (
                torch.rand(self.N, device=x.device) <= scores.exp()
            )  # h(u) = min(1, u)

            x[accept, :] = y[accept, :]  # MCMC update

        self.x = x

        # Update potential
        self.V = lambda x: (1 - a) * self.V0(x) + a * self.distribution.potential(x)

        info = {"sample": x, "ESS": ESS}

        return info


# Our first CMC method ============================================


class CMC(MonteCarloSampler):
    """Collective Monte-Carlo."""

    def __init__(self, space, start, proposal, annealing=None, verbose=False):
        super().__init__(space, start, proposal, verbose=verbose)
        self.annealing = annealing

    def sample_proposal(self, x):

        N = len(x)
        indices = torch.randint(N, size=(N,)).to(x.device)
        return self.proposal.sample(x[indices, :])  # Proposal

    def proposal_potential(self, x, ratio):
        y = self.sample_proposal(x)

        V_x, Prop_x = self.distribution.potential(x), self.proposal.potential(x)(x)
        V_y, Prop_y = self.distribution.potential(y), self.proposal.potential(x)(y)

        return y, V_x, V_y, Prop_x, Prop_y

    def update_kernel(self, scores):
        None

    def extra_info(self):
        return {}

    def update(self):
        x = self.x
        N = len(x)

        # Annealing ratio
        ratio = (
            1
            if self.annealing is None
            else 1 - np.exp(-self.iteration / self.annealing)
        )

        # Get the proposal:
        y, V_x, V_y, Prop_x, Prop_y = self.proposal_potential(x, ratio)

        # Logarithm of the acceptance ratio:
        scores = ratio * (V_x - V_y) + Prop_y - Prop_x
        accept = torch.rand(N).type_as(x) <= scores.exp()  # h(u) = min(1, u)
        x[accept, :] = y[accept, :]  # MCMC update

        self.update_kernel(scores)  # update the kernel probabilities
        rate = (1.0 * accept).mean()
        # x = x.clamp(0, 1)  # Clip to the unit square
        self.x = x

        # Recompute the volume of the ball in order to count the neighbours

        volumes = float(
            np.pi ** (self.proposal.D / 2) / gamma(self.proposal.D / 2 + 1)
        ) * (torch.tensor(self.proposal.s, device=self.x.device) ** self.proposal.D)

        # (-Prop_y + volumes.log() + np.log(len(x))).exp().mean(0)

        info = {
            "sample": x,
            "proposal": y,
            "rate": rate,
            "normalizing constant": (Prop_y - V_y).exp().mean(),
            "number of neighbours": (-Prop_y).exp().mean(0) * volumes * float(len(x)),
        }
        info = {**info, **self.extra_info()}

        return info


class Ada_CMC(CMC):
    """Collective Monte-Carlo, with adaptive anisotropic kernels."""

    def proposal_potential(self, x, ratio):
        N = len(x)
        indices = torch.randint(N, size=(N,)).to(x.device)
        self.proposal.adapt(x)
        y = self.proposal.adaptive_sample(x, indices)  # Proposal

        V_x, Prop_x = self.distribution.potential(x), self.proposal.potential(x)(x)
        V_y, Prop_y = self.distribution.potential(y), self.proposal.potential(x)(y)

        return y, V_x, V_y, Prop_x, Prop_y


class MOKA_Markov_CMC(CMC):
    """Collective Monte-Carlo, with kernel weights that are adapted at every step."""

    def proposal_potential(self, x, ratio):
        N = len(x)

        # Update the kernel probabilities ----------------------
        # Evaluate the potential on the sample, and normalize it
        V_x = self.distribution.potential(x)

        logpi_x = -V_x
        logpi_x -= logpi_x.logsumexp(-1)  # (N,)

        # Evaluate the proposal kernels on the sample, and normalize them
        logprop_x = -self.proposal.nlog_densities(x)
        logprop_x -= logprop_x.logsumexp(0, keepdim=True)  # (N, K)

        # Get the current vector of kernel weights
        probas = torch.ones_like(self.proposal.probas)  # .clone()
        probas = probas / probas.sum()

        log_probas = probas.log().detach()
        log_probas.requires_grad = True
        # optimizer = torch.optim.LBFGS([log_probas], line_search_fn = "strong_wolfe")
        optimizer = torch.optim.Adam([log_probas], lr=1)

        # Define the auxiliary function to optimize
        def closure():
            optimizer.zero_grad()
            log_probas_normalized = log_probas - log_probas.logsumexp(-1)
            log_products = logprop_x + log_probas_normalized.view(1, -1)
            logprobas_x = log_products.logsumexp(-1)

            # Total Variation penalty:
            loss = (logprobas_x.exp() - logpi_x.exp()).abs().sum()

            loss.backward()
            return loss

        # Optimization loop
        for it in range(100):
            optimizer.step(closure)

        # Update the probas:
        log_probas_normalized = log_probas - log_probas.logsumexp(-1)
        probas = log_probas_normalized.exp()
        self.proposal.probas = probas.detach()

        # Proposal and potential -------------------------------
        y = self.sample_proposal(x)  # Proposal

        Prop_x = self.proposal.potential(x)(x)
        V_y, Prop_y = self.distribution.potential(y), self.proposal.potential(x)(y)

        return y, V_x, V_y, Prop_x, Prop_y

    def extra_info(self):
        return {"probas": self.proposal.probas}


class MOKA_CMC(CMC):
    """Collective Monte-Carlo, with adaptive kernel weights."""

    def sample_proposal(self, x):
        N = len(x)
        indices = torch.randint(N, size=(N,)).to(x.device)
        y, scale_indices = self.proposal.sample_indices(x[indices, :])
        self.scale_indices = scale_indices
        return y

    def update_kernel(self, scores):
        # Bound the scores from below (more stable)
        scores[scores < -100] = -100
        # Update the kernel probabilities:
        probas = self.proposal.probas.clone()
        avg_score = self.proposal.probas.clone()
        for i in range(len(probas)):
            # probas[i] = scores[accept & (scale_indices == i)].exp().sum()
            scores_i = scores[self.scale_indices == i]
            if len(scores_i) == 0:
                avg_score[i] = -1000000.0  # -> probas[i] = 0. -> probas[i] = 1%
            else:
                avg_score[i] = scores_i.mean()

        # Normalize, go back to "normal" domain
        avg_score = avg_score - avg_score.logsumexp(0)
        probas = avg_score.exp()

        # Minimal probability per scale: 1%
        probas[probas < 0.01] = 0.01  # Otherwise, some scales may disappear
        # Don't forget to re-normalize
        probas = probas / probas.sum()

        if torch.isnan(probas.sum()):
            print("WARNING! Some NaN in the kernel probas have been removed.")
            print(scores)
            probas = torch.ones(len(probas), device=probas.device) / len(probas)

        self.proposal.probas = probas

    def extra_info(self):
        return {"probas": self.proposal.probas}


from scipy import stats


class MOKA_KIDS_CMC(MOKA_CMC):
    """Kernel Importance-by-Deconvolution Sampling Collective Monte-Carlo."""

    def __init__(
        self, space, start, proposal, annealing=None, verbose=False, iterations=100
    ):
        super().__init__(space, start, proposal, verbose=verbose, annealing=annealing)
        self.nits = iterations

    def proposal_potential(self, x, ratio):
        N = len(x)
        V_x = self.distribution.potential(x)

        # Richardson-Lucy-like iterations ----------------------------------
        # We look for u such that
        #   proposal.potential(x, u)(x_i) = - log( k * e^u ) (x_i) = q * V(x_i)

        target = -ratio * V_x
        target = target - target.logsumexp(0)  # Normalize the target log-likelihood

        # u = (- np.log(N) * np.ones(N)).astype(dtype)
        u = target
        for it in range(self.nits):
            offset = target + self.proposal.potential(x, u)(x)
            offset = -self.proposal.potential(x, offset)(
                x
            )  #  Genuine Richardson-Lucy would have this line too
            offset[torch.isnan(offset)] = torch.zeros(
                torch.isnan(offset).sum(), device=offset.device
            )
            offset[offset > 1e3] = 1e3 * torch.ones(
                (offset > 1e3).sum(), device=offset.device
            )
            u = u + offset

        u = u - u.logsumexp(0)  # Normalize the proposal
        # print(stats.describe(numpy(N * u.exp())))

        # Importance sampling: ---------------------------------------------
        indices = np.random.choice(N, size=N, p=numpy(u.exp()))
        indices = torch.from_numpy(indices).to(x.device)
        y, scale_indices = self.proposal.sample_indices(x[indices, :])  # Proposal
        self.scale_indices = scale_indices

        # Potentials:
        # -------------------------------------------
        V_y = self.distribution.potential(y)
        Prop_x = self.proposal.potential(x, u)(x)
        Prop_y = self.proposal.potential(x, u)(y)
        self.u = u  # Save for later plot

        return y, V_x, V_y, Prop_x, Prop_y

    def extra_info(self):
        return {
            "log-weights": self.u,
            "probas": self.proposal.probas,
        }


class KIDS_CMC(MOKA_KIDS_CMC):
    """Kernel Importance-by-Deconvolution Sampling Collective Monte-Carlo."""

    # We implement "KIDS" as "MOKA_KIDS", without kernel updates.
    def update_kernel(self, scores):
        None

    def extra_info(self):
        return {"log-weights": self.u}

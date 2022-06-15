# GPU accelerated collective Monte Carlo methods

This folder contains the code for the arXiv preprint [Collective Proposal Distributions for Nonlinear MCMC samplers: Mean-Field Theory and Fast Implementation](https://arxiv.org/pdf/1909.08988.pdf).

Please visit our [website](http://kernel-operations.io/monaco/) for a full documentation.

# Examples 

The target distribution is a mixture of a banana-shaped distribution and three Gaussian distributions in dimension 2. The level sets of the target distribution are shown in red, the N particles in blue and the (rejected) proposals in green. 

 - The Vanilla CMC algorithm. 

https://user-images.githubusercontent.com/70896255/173736027-68039e97-3560-491e-938d-e972969cd8f6.mp4

- The MoKA Markov algorithm (adaptive version with a mixture of proposal distributions with different sizes updated at each iteration).

https://user-images.githubusercontent.com/70896255/173736058-d59da4a0-2fe5-4e1a-9d57-8e6d2e88ab89.mp4

- The non-Markovian MoKA algorithm (adaptive version with a mixture of proposal distributions with different sizes updated at each iteration and depending on the past iterations).

https://user-images.githubusercontent.com/70896255/173736069-84c89eca-5238-4048-80e7-1584fcc77f2c.mp4

- The non-Markovian MoKA algorithm with the KIDS weighting procedure in order to select the best particles. 

https://user-images.githubusercontent.com/70896255/173736077-2ede28dc-7327-40c3-ad5a-97cffda17bca.mp4

- The classical Metropolis-Hastings algorithm with N independent chains and a large proposal distribution. 

https://user-images.githubusercontent.com/70896255/173736087-2d2cdf23-3281-4ba9-b06b-2d4951601671.mp4

More examples in the article! 

# Authors

- [Antoine Diez](https://antoinediez.gitlab.io)
- [Grégoire Clarté](https://www.ceremade.dauphine.fr/~clarte/)
- [Jean Feydy](https://www.jeanfeydy.com)

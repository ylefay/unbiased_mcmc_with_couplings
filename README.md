# unbiased_mcmc_with_couplings

Implementation of "Unbiased Markov-Chain Monte Carlo estimators with couplings". Project done as part of the Bayesian
Machine Learning course by Rémi Bardenet and Julyan Arbel.

It includes a coupled Metropolis-Hasting algorithm, the maximal coupling and its generalization, the modified Thorisson's Algorithm.

The R companion-code of the paper is available
at [https://github.com/pierrejacob/unbiasedmcmc](https://github.com/pierrejacob/unbiasedmcmc).

# How?
This package requires `jax` and `numpy`. You can use `pip install -e .` to install the package.
See the `experiments` folder for some examples of how to use the package. 

# References

- Pierre E. Jacob, John O'Leary, Yves F. Atchadé, "[Unbiased Markov-Chain Monte Carlo estimators with couplings](https://academic.oup.com/jrsssb/article/82/3/543/7056129)", 2020
- Yves F. Atchadé, Pierre E. Jacob, "[Unbiased Markov Chain Monte Carlo: what, why and how](https://math.bu.edu/people/atchade/umcmc_rev.pdf)"

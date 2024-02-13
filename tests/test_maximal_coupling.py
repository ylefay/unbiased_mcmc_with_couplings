from pymcmc_unbiased.distrib.proposal import random_walk_mh_proposal
from pymcmc_unbiased.metropolis_hasting import *
from pymcmc_unbiased.utils import reconstruct_chains, get_coupling_time
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from functools import partial
import matplotlib.pyplot as plt

from random import randint

def normal_logpdf(x, mu, chol_sigma):
    sigma = chol_sigma @ chol_sigma.T
    return multivariate_normal.logpdf(x, mean=mu, cov=sigma)

def test_maximal_coupling():
    OP_key = jax.random.PRNGKey(randint(0, 1<<30))
    key, next_key = jax.random.split(OP_key, 2)

    dim = 1
    lag = 3

    chol_sigma = jnp.eye(dim) * 0.1
    q_hat = partial(random_walk_mh_proposal, chol_sigma=chol_sigma)

    def log_q(xp, x):
        return normal_logpdf(x=xp, mu=x, chol_sigma=chol_sigma)

    def log_target(x):
        return normal_logpdf(x=x, mu=jnp.ones(dim), chol_sigma=jnp.eye(dim))

    x0y0 = jax.random.uniform(key, shape=(2 * dim,))
    x0 = x0y0.at[:dim].get()
    y0 = x0y0.at[dim:].get()

    N = 17
    keys = jax.random.split(next_key, N)
    Xs_before_lag, chains, tau = mh_maximal_coupling_with_lag(keys, x0=x0, y0=y0, q_hat=q_hat, log_q=log_q,
                                                              log_target=log_target, lag=lag)
    print(tau)
    reconstructed = reconstruct_chains(Xs_before_lag, chains)
    tau = get_coupling_time(reconstructed)
    print(tau)
    Xs, Ys = reconstructed
    plt.plot(range(len(Xs)), Xs, label='Xs')
    plt.plot(range(lag, lag + len(Ys)), Ys, label='Ys')
    plt.savefig("output.png")
    return Xs, Ys


if __name__ == '__main__':
    test_maximal_coupling()
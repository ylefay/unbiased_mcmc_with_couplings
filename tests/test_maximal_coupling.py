from src.pymcmc_unbiased.distrib.proposal import random_walk_mh_proposal
from src.pymcmc_unbiased.distrib.logpdf import normal_logpdf
from src.pymcmc_unbiased.mh import *
import jax
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt


def test_maximal_coupling():
    OP_key = jax.random.PRNGKey(0)
    key, next_key = jax.random.split(OP_key, 2)

    dim = 1
    lag = 10

    chol_sigma = jnp.eye(dim) * 0.1
    q_hat = partial(random_walk_mh_proposal, chol_sigma=chol_sigma)

    def log_q(xp, x):
        return normal_logpdf(x=xp, mu=x, chol_sigma=chol_sigma)

    def log_target(x):
        return normal_logpdf(x=x, mu=jnp.ones(dim), chol_sigma=jnp.eye(dim))

    x0y0 = jax.random.uniform(key, shape=(2 * dim,))
    x0 = x0y0.at[:dim].get()
    y0 = x0y0.at[dim:].get()

    N = 100
    keys = jax.random.split(next_key, N)
    Xs_before_lag, chains = metropolis_hasting_maximal_coupling_with_lag(keys, x0=x0, y0=y0, q_hat=q_hat, log_q=log_q,
                                                             log_target=log_target, lag=lag)
    plt.plot(chains[0])
    plt.plot(chains[1])
    print(Xs_before_lag)
    print(chains[0])
    print(chains[1])
    print(len(chains[0]))
    print(len(chains[1]))
    print(len(Xs_before_lag))
    plt.show()
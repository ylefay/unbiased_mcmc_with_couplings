from pymcmc_unbiased.distrib.proposal import random_walk_mh_proposal
from pymcmc_unbiased.distrib.logpdf import normal_logpdf
from pymcmc_unbiased.metropolis_hasting import *
from pymcmc_unbiased.utils import reconstruct_chains, get_coupling_time
import jax
import jax.numpy as jnp
from functools import partial


@jax.vmap
def maximal_coupling(key):
    key, next_key = jax.random.split(key, 2)

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
    reconstructed = reconstruct_chains(Xs_before_lag, chains)
    tau = get_coupling_time(reconstructed)
    Xs, Ys = reconstructed
    return Xs, Ys, tau


if __name__ == '__main__':
    OP_key = jax.random.PRNGKey(0)
    KEYS = jax.random.split(OP_key, 2)
    print(maximal_coupling(KEYS))

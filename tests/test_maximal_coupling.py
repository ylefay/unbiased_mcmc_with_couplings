import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from functools import partial
import matplotlib.pyplot as plt
from random import randint

from pymcmc_unbiased.chain import run_chain_coupled


def random_walk_mh_proposal(key, x, chol_sigma):
    return x + chol_sigma @ jax.random.normal(key, (x.shape[-1],))


def normal_logpdf(x, mu, chol_sigma):
    sigma = chol_sigma @ chol_sigma.T
    return multivariate_normal.logpdf(x, mean=mu, cov=sigma)


def test():
    OP_key = jax.random.PRNGKey(randint(0, 1 << 30))
    chain_key, x0_key, y0_key = jax.random.split(OP_key, 3)

    n_chain = 150
    dim = 1
    lag = 10

    print(f"chain length: {n_chain}, dim x: {dim}, lag: {lag}")

    x0 = jax.random.uniform(x0_key, shape=(dim,))
    y0 = jax.random.uniform(y0_key, shape=(dim,))

    chol_sigma = jnp.eye(dim) * 0.1
    q_hat = partial(random_walk_mh_proposal, chol_sigma=chol_sigma)

    def log_q(xp, x):
        return normal_logpdf(x=xp, mu=x, chol_sigma=chol_sigma)

    def log_target(x):
        return normal_logpdf(x=x, mu=jnp.ones(dim), chol_sigma=jnp.eye(dim))

    is_coupled, time_coupled, Xs, Ys = run_chain_coupled(
        key=chain_key,
        x0=x0,
        y0=y0,
        q_hat=q_hat,
        log_q=log_q,
        log_target=log_target,
        lag=lag,
        dim=dim,
        n_chain=n_chain
    )

    print(f"is coupled: {is_coupled}, time coupled: {time_coupled}")

    plt.plot(range(len(Xs)), Xs, label='Xs')
    plt.plot(range(lag, lag + len(Ys)), Ys, label='Ys')
    plt.legend(loc="best")
    plt.savefig("output.png")

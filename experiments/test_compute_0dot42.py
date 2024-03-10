from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal

from pymcmc_unbiased.monte_carlo_estimators import default_monte_carlo_estimator, unbiased_monte_carlo_estimation

OP_key = jax.random.PRNGKey(0)


def random_walk_mh_proposal(key, x, chol_sigma):
    return x + chol_sigma @ jax.random.normal(key, (x.shape[-1],))


@jax.vmap
def simulation_default(key):
    n_chain = 100000
    dim = 1
    chain_key, x0_key = jax.random.split(key, 2)

    def pi_0(key):
        return jax.random.multivariate_normal(key, mean=4 * jnp.ones(dim, ), cov=1 * jnp.eye(dim, ))

    x0 = pi_0(x0_key)

    chol_sigma = jnp.eye(dim) * 3.0
    q_hat = partial(random_walk_mh_proposal, chol_sigma=chol_sigma)

    def log_q(xp, x):
        return multivariate_normal.logpdf(xp, mean=x, cov=chol_sigma @ chol_sigma.T)

    def log_target(x):
        p = 0.5
        mu = 4. * jnp.ones(dim)
        chol_sigma = jnp.eye(dim)
        return jnp.log(p * multivariate_normal.pdf(x, mean=mu, cov=chol_sigma @ chol_sigma.T) +
                       (1 - p) * multivariate_normal.pdf(x, mean=-mu, cov=chol_sigma @ chol_sigma.T))

    def h(x):
        return jax.lax.cond(x[0] > 3., lambda _: 1.0, lambda _: 0.0, x)

    return default_monte_carlo_estimator(chain_key, h, x0, q_hat, log_q, log_target, n_chain)


@jax.vmap
def simulation_unbiased(key):
    k = 100
    m = 10000
    dim = 1
    lag = 1

    chain_key, x0_key, y0_key = jax.random.split(key, 3)

    def pi_0(key):
        return jax.random.multivariate_normal(key, mean=4 * jnp.ones(dim, ), cov=1 * jnp.eye(dim, ))

    x0 = pi_0(x0_key)
    y0 = pi_0(y0_key)

    chol_sigma = jnp.eye(dim) * 3.0
    q_hat = partial(random_walk_mh_proposal, chol_sigma=chol_sigma)

    def log_q(xp, x):
        return multivariate_normal.logpdf(x=xp, mean=x, cov=chol_sigma @ chol_sigma.T)

    def log_target(x):
        p = 0.5
        mu = 4. * jnp.ones(dim)
        chol_sigma = jnp.eye(dim) * 1.0
        return jnp.log(p * multivariate_normal.pdf(x, mean=mu, cov=chol_sigma @ chol_sigma.T) +
                       (1 - p) * multivariate_normal.pdf(x, mean=-mu, cov=chol_sigma @ chol_sigma.T))

    def h(x):
        return jax.lax.cond(x[0] > 3., lambda _: 1.0, lambda _: 0.0, x)

    return unbiased_monte_carlo_estimation(chain_key, h, x0, y0, q_hat, log_q, log_target, lag, k, m)


def test():
    keys = jax.random.split(OP_key, 1_000)

    _ = simulation_default(keys)
    print(_.mean())
    print("simulation default done!")

    _, is_coupled, time, meeting_time = simulation_unbiased(keys)
    print(_.mean())
    print("simulation unbiased done!")
    print(is_coupled.flatten())
    print(time.flatten())

import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from functools import partial
from random import randint

from pymcmc_unbiased.monte_carlo_estimators import default_monte_carlo_estimator, unbiased_monte_carlo_estimation

OP_key = jax.random.PRNGKey(0)


def random_walk_mh_proposal(key, x, chol_sigma):
    return x + chol_sigma @ jax.random.normal(key, (x.shape[-1],))


def normal_logpdf(x, mu, chol_sigma):
    sigma = chol_sigma @ chol_sigma.T
    return multivariate_normal.logpdf(x, mean=mu, cov=sigma)


@jax.vmap
@jax.jit
def simulation_default(key):
    n_chain = 1000
    dim = 2
    chain_key, x0_key = jax.random.split(key, 2)
    x0 = jax.random.uniform(x0_key, shape=(dim,))

    chol_sigma = jnp.eye(dim)
    q_hat = partial(random_walk_mh_proposal, chol_sigma=chol_sigma)

    def log_q(xp, x):
        return normal_logpdf(x=xp, mu=x, chol_sigma=chol_sigma)

    def log_target(x):
        return jax.lax.cond((x[0] <= 1.) & (x[1] <= 1.) & (x[0] >= 0.) & (x[1] >= 0.), lambda _: 1.,
                            lambda _: -50.,
                            None)

    def h(x):
        return jax.lax.cond(jnp.linalg.norm(x, ord=2) ** 2 <= 1, lambda: 1., lambda: 0.)

    return default_monte_carlo_estimator(chain_key, h, x0, q_hat, log_q, log_target, n_chain)


@jax.vmap
@jax.jit
def simulation_unbiased(key):
    k = 100
    m = 1000
    dim = 2
    lag = 1

    chain_key, x0_key, y0_key = jax.random.split(key, 3)

    x0 = jax.random.uniform(x0_key, shape=(dim,))
    y0 = jax.random.uniform(y0_key, shape=(dim,))

    chol_sigma = jnp.eye(dim)
    q_hat = partial(random_walk_mh_proposal, chol_sigma=chol_sigma)

    def log_q(xp, x):
        return normal_logpdf(x=xp, mu=x, chol_sigma=chol_sigma)

    def log_target(x):
        return jax.lax.cond((x[0] <= 1.) & (x[1] <= 1.) & (x[0] >= 0.) & (x[1] >= 0.), lambda _: 1.,
                            lambda _: -50.,
                            None)

    def h(x):
        return jax.lax.cond(jnp.linalg.norm(x, ord=2) ** 2 <= 1, lambda: 1., lambda: 0.)

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

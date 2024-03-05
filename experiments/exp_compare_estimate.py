# Here we compare the performance of the unbiased estimator with the default estimator
# on a bimodal 1D target distribution. Trying to replicate the experiment 5.1.

import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from random import randint

from pymcmc_unbiased.monte_carlo_estimators import (
    default_monte_carlo_estimator,
    unbiased_monte_carlo_estimation,
)


def random_walk_mh_proposal(key, x, chol_sigma):
    return x + chol_sigma @ jax.random.normal(key, (x.shape[-1],))


def normal_logpdf(x, mu, chol_sigma):
    sigma = chol_sigma @ chol_sigma.T
    return multivariate_normal.logpdf(x, mean=mu, cov=sigma)


def log_target_builder(p, mu, chol_sigma):
    """
    Bimodal normal distribution:
        pN(m, s^2) + (1-p)N(-m, s^2)
    """

    def log_target(x):
        return jnp.log(p * multivariate_normal.pdf(x, mean=mu, cov=chol_sigma @ chol_sigma.T) +
                       (1 - p) * multivariate_normal.pdf(x, mean=-mu, cov=chol_sigma @ chol_sigma.T))

    return log_target


@jax.vmap
def simulation_default(key):
    n_chain = 100000
    chain_key, x0_key = jax.random.split(key, 2)

    def pi_0(key):
        return jax.random.multivariate_normal(key, mean=4 * jnp.ones(dim, ), cov=1 * jnp.eye(dim, ))

    x0 = pi_0(x0_key)

    chol_sigma = jnp.eye(dim) * 3.0
    q_hat = partial(random_walk_mh_proposal, chol_sigma=chol_sigma)

    def log_q(xp, x):
        return normal_logpdf(x=xp, mu=x, chol_sigma=chol_sigma)

    return default_monte_carlo_estimator(
        chain_key, h, x0, q_hat, log_q, log_target, n_chain
    )


def simulation_unbiased_builder(k, m):
    @jax.vmap
    def simulation_unbiased(key):
        chain_key, x0_key, y0_key = jax.random.split(key, 3)

        def pi_0(key):
            return jax.random.multivariate_normal(key, mean=4 * jnp.ones(dim, ), cov=1 * jnp.eye(dim, ))

        x0 = pi_0(x0_key)
        y0 = pi_0(y0_key)

        chol_sigma = jnp.eye(dim) * 3.0
        q_hat = partial(random_walk_mh_proposal, chol_sigma=chol_sigma)

        def log_q(xp, x):
            return normal_logpdf(x=xp, mu=x, chol_sigma=chol_sigma)

        return unbiased_monte_carlo_estimation(
            chain_key, h, x0, y0, q_hat, log_q, log_target, lag, k, m
        )

    return simulation_unbiased


if __name__ == "__main__":

    dim = 1
    n_samples = 1_000

    ks = [1, 100, 200]
    m_mults = [1, 10, 20]
    lag = 1

    chol_sigma_target = 1.0 * jnp.eye(dim)
    mu = jnp.ones(dim) * 4.0
    log_target = log_target_builder(0.5, mu, chol_sigma_target)


    def h(x):
        return jax.lax.cond(x[0] > 3., lambda _: 1.0, lambda _: 0.0, x)


    OP_key = jax.random.PRNGKey(randint(0, 1 << 30))

    df = pd.DataFrame(
        columns=[
            "k",
            "m",
            "ratio_coupled",
            "mean_estimate_unbiased",
            "mean_estimate_biased",
            "variance_unbiased",
            "variance_default",
            "average_time",
        ]
    )

    for k in tqdm(ks):
        for m_mult in m_mults:
            m = k * m_mult
            OP_key, new_key = jax.random.split(OP_key)
            keys = jax.random.split(new_key, n_samples)

            samples_unbiased, is_coupled, time, meeting_time = (
                simulation_unbiased_builder(k, m)(keys)
            )
            samples = simulation_default(keys)
            df.loc[len(df)] = {
                "k": k,
                "m": m,
                "ratio_coupled": sum(is_coupled) / n_samples,
                "mean_estimate_unbiased": jnp.mean(samples_unbiased),
                "mean_estimate_biased": jnp.mean(samples),
                "variance_unbiased": jnp.var(samples_unbiased),
                "variance_default": jnp.var(samples),
                "average_time": jnp.mean(time),
            }

    df.to_csv("exp_compare_estimate.csv")

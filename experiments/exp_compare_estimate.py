# Here we compare the performance of the unbiased estimator with the default estimator
# on a bimodal 1D target distribution. Trying to replicate the experiment 5.1.
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from functools import partial
import pickle

from pymcmc_unbiased.monte_carlo_estimators import (
    default_monte_carlo_estimator,
    unbiased_monte_carlo_estimation,
)

OP_key = jax.random.PRNGKey(0)


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


def h(x):
    """
    Test function.
    The integral should be ~0.42.
    """
    return jax.lax.cond(x[0] > 3., lambda _: 1.0, lambda _: 0.0, x)


def pi_0(key):
    """
    Initial distribution
    """
    return jax.random.multivariate_normal(key, mean=4 * jnp.ones(dim, ), cov=1 * jnp.eye(dim, ))


@jax.jit
def simulation_default(key, burnin_period, n_chain):
    chain_key, x0_key = jax.random.split(key, 2)

    x0 = pi_0(x0_key)

    chol_sigma = jnp.eye(dim) * 3.0
    q_hat = partial(random_walk_mh_proposal, chol_sigma=chol_sigma)

    def log_q(xp, x):
        return normal_logpdf(x=xp, mu=x, chol_sigma=chol_sigma)

    return default_monte_carlo_estimator(chain_key, h, x0, q_hat, log_q, log_target, n_chain, burnin_period)


@jax.jit
def simulation_unbiased(key, k, m, lag):
    chain_key, x0_key, y0_key = jax.random.split(key, 3)

    x0 = pi_0(x0_key)
    y0 = pi_0(y0_key)

    chol_sigma = jnp.eye(dim) * 3.0
    q_hat = partial(random_walk_mh_proposal, chol_sigma=chol_sigma)

    def log_q(xp, x):
        return normal_logpdf(x=xp, mu=x, chol_sigma=chol_sigma)

    return unbiased_monte_carlo_estimation(chain_key, h, x0, y0, q_hat, log_q, log_target, lag, k, m)


if __name__ == "__main__":

    result = dict()

    dim = 1
    n_samples = 1_0000

    ks = [1, 100, 200]
    m_mults = [1, 10, 100]
    lags = [1, 10, 100]

    chol_sigma_target = 1.0 * jnp.eye(dim)
    mu = jnp.ones(dim) * 4.0
    log_target = log_target_builder(0.5, mu, chol_sigma_target)

    for k in ks:
        result[k] = dict()
        for m_mult in m_mults:
            m = k * m_mult
            result[k][m] = dict()
            OP_key, key, key2 = jax.random.split(OP_key, 3)
            keys = jax.random.split(key, n_samples)
            samples = jax.vmap(simulation_default, in_axes=(0, None, None))(keys, k, m - k + 1)
            for lag in lags:
                print(k, m, lag)
                if lag <= k:
                    result[k][m][lag] = dict()
                    key2, new_key2 = jax.random.split(key2, 2)
                    keys = jax.random.split(new_key2, n_samples)
                    samples_unbiased, is_coupled, time, meeting_time = jax.vmap(simulation_unbiased,
                                                                                in_axes=(0, None, None, None))(keys, k,
                                                                                                               m, lag)
                    result[k][m][lag] = [samples, samples_unbiased, is_coupled, time, meeting_time]

    with open("exp_compare_estimators.pkl", "wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

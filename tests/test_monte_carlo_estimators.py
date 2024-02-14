import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from functools import partial
import matplotlib.pyplot as plt
from random import randint

from pymcmc_unbiased.monte_carlo_estimators import default_monte_carlo_estimator, unbiased_monte_carlo_estimation

def random_walk_mh_proposal(key, x, chol_sigma):
    return x + chol_sigma @ jax.random.normal(key, (x.shape[-1],))

def normal_logpdf(x, mu, chol_sigma):
    sigma = chol_sigma @ chol_sigma.T
    return multivariate_normal.logpdf(x, mean=mu, cov=sigma)

@jax.vmap
def simulation_default(key):
    n_chain = 1000
    dim = 1
    chain_key, x0_key = jax.random.split(key, 2)
    x0 = jax.random.uniform(x0_key, shape=(dim,))

    chol_sigma = jnp.eye(dim) * 0.1
    q_hat = partial(random_walk_mh_proposal, chol_sigma=chol_sigma)

    def log_q(xp, x):
        return normal_logpdf(x=xp, mu=x, chol_sigma=chol_sigma)

    def log_target(x):
        return normal_logpdf(x=x, mu=jnp.ones(dim), chol_sigma=jnp.eye(dim))
    
    def h(x):
        return 1.
    
    return default_monte_carlo_estimator(chain_key, h, x0, q_hat, log_q, log_target, n_chain)

@jax.vmap
def simulation_unbiased(key):
    k = 30
    m = 150
    dim = 1
    lag = 10
    
    chain_key, x0_key, y0_key = jax.random.split(key, 3)
    
    x0 = jax.random.uniform(x0_key, shape=(dim,))
    y0 = jax.random.uniform(y0_key, shape=(dim,))

    chol_sigma = jnp.eye(dim) * 0.1
    q_hat = partial(random_walk_mh_proposal, chol_sigma=chol_sigma)

    def log_q(xp, x):
        return normal_logpdf(x=xp, mu=x, chol_sigma=chol_sigma)

    def log_target(x):
        return normal_logpdf(x=x, mu=jnp.ones(dim), chol_sigma=jnp.eye(dim))
    
    def h(x):
        return 1.
    
    return unbiased_monte_carlo_estimation(chain_key, h, x0, y0, q_hat, log_q, log_target, lag, k, m)

def main():
    OP_key = jax.random.PRNGKey(randint(0, 1<<30))
    keys = jax.random.split(OP_key, 50)

    simulation_default(keys)
    print("simulation default done!")

    _, is_coupled, time = simulation_unbiased(keys)
    print("simulation unbiased done!")
    print(is_coupled.flatten())
    print(time.flatten())


if __name__ == '__main__':
    main()
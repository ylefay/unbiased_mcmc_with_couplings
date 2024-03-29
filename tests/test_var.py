from functools import partial
from random import randint

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.stats import multivariate_normal

from pymcmc_unbiased.monte_carlo_estimators import default_monte_carlo_estimator, unbiased_monte_carlo_estimation


def random_walk_mh_proposal(key, x, chol_sigma):
    return x + chol_sigma @ jax.random.normal(key, (x.shape[-1],))
 
 
def normal_logpdf(x, mu, chol_sigma):
    sigma = chol_sigma @ chol_sigma.T
    return multivariate_normal.logpdf(x, mean=mu, cov=sigma)
 
 
dim = 2
# defining the proposal distribution
chol_sigma = jnp.eye(dim) * 0.1
q_hat = partial(random_walk_mh_proposal, chol_sigma=chol_sigma)
 
 
def log_q(xp, x):
    return normal_logpdf(x=xp, mu=x, chol_sigma=chol_sigma)
 
 
# defining the target distribution
def log_target(x):
    return jax.lax.cond((x[0] <= 1.) & (x[1] <= 1.) & (x[0] >= 0.) & (x[1] >= 0.), lambda _: 1.,
                        lambda _: -50.,
                        None)
 
 
# defining the test function
def h(x):
    return jax.lax.cond(jnp.linalg.norm(x, ord=2) ** 2 <= 1, lambda: 1., lambda: 0.)
 
 
@partial(jax.vmap, in_axes=(0, None))
def simulation_default(key, m):
    n_chain = m
    chain_key, x0_key = jax.random.split(key, 2)
    x0 = jax.random.uniform(x0_key, shape=(dim,))
 
    return default_monte_carlo_estimator(chain_key, h, x0, q_hat, log_q, log_target, n_chain)
 
 
@partial(jax.vmap, in_axes=(0, None))
def simulation_unbiased(key, t):
    m, k = t
    lag = 10
 
    chain_key, x0_key, y0_key = jax.random.split(key, 3)
 
    x0 = jax.random.uniform(x0_key, shape=(dim,))
    y0 = jax.random.uniform(y0_key, shape=(dim,))
 
    return unbiased_monte_carlo_estimation(chain_key, h, x0, y0, q_hat, log_q, log_target, lag, k, m)
 
 
def experiment(k, m, N=1000):
    OP_key = jax.random.PRNGKey(randint(0, 1 << 30))
    keys = jax.random.split(OP_key, N)
 
    simple_mcmc_estimator = simulation_default(keys, m)
    print(f"k: {k}, m: {m}")
    print(simple_mcmc_estimator.mean())
    print("simulation default done!")
 
    coupled_mcmc_estimator, is_coupled, time = simulation_unbiased(keys, (m, k))
    print(coupled_mcmc_estimator.mean())
    print("simulation unbiased done!")
    #print(is_coupled.flatten())
    #print(time.flatten())
    return np.array([coupled_mcmc_estimator.mean(), coupled_mcmc_estimator.std(), simple_mcmc_estimator.mean(),
                     simple_mcmc_estimator.std()])
 
 
if __name__ == "__main__":
    MK = [(0, 10), (50, 100), (100, 500), (500, 1000), (1000, 5000), (5000, 10000)]
    res = np.zeros(shape=(len(MK), 4))

    for i, (k, m) in enumerate(MK):
        res[i] = experiment(k, m)

    plt.errorbar([t[1] for t in MK], res[:, 0], yerr=res[:, 1], label="coupled_mcmc_estimator_mean")
    plt.errorbar([t[1] for t in MK], res[:, 2], yerr=res[:, 3], label="simple_mcmc_estimator_mean", alpha=0.8)
    plt.legend()
    plt.savefig("result.png")
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from random import randint

from pymcmc_unbiased.monte_carlo_estimators import default_monte_carlo_estimator, unbiased_monte_carlo_estimation


def random_walk_mh_proposal(key, x, chol_sigma):
    return x + chol_sigma @ jax.random.normal(key, (x.shape[-1],))


def normal_logpdf(x, mu, chol_sigma):
    sigma = chol_sigma @ chol_sigma.T
    return multivariate_normal.logpdf(x, mean=mu, cov=sigma)


@jax.vmap
def simulation_unbiased(key):
    k = 100
    m = 10*k
    lag = 2

    chain_key, x0_key, y0_key = jax.random.split(key, 3)

    x0 = jax.random.uniform(x0_key, shape=(dim,))
    y0 = jax.random.uniform(y0_key, shape=(dim,))

    chol_sigma = jnp.eye(dim) * 0.1
    q_hat = partial(random_walk_mh_proposal, chol_sigma=chol_sigma)

    def log_q(xp, x):
        return normal_logpdf(x=xp, mu=x, chol_sigma=chol_sigma)
    
    def h(x):
        return 1.

    return unbiased_monte_carlo_estimation(chain_key, h, x0, y0, q_hat, log_q, log_target, lag, k, m)


if __name__ == "__main__":

    dims = [1, 2, 3, 4]
    n_samples = 100

    df = pd.DataFrame(columns=["dim", "time", "is_coupled"])


    def log_target_builder(mu, chol_sigma):
        def log_target(x):
            return multivariate_normal.logpdf(x, mean=mu, cov=chol_sigma @ chol_sigma.T)
        return log_target
    

    OP_key = jax.random.PRNGKey(randint(0, 1 << 30))
    key_dims = jax.random.split(OP_key, len(dims))

    for dim, key in tqdm(zip(dims, key_dims)):
        keys = jax.random.split(key, n_samples)
        chol_sigma = jnp.eye(dim) * 0.1
        mu = jnp.ones(dim)
        log_target = log_target_builder(mu, chol_sigma)
        samples_unbiased, is_coupled, time = simulation_unbiased(keys)
        for i in range(n_samples):
            df.loc[len(df)] = {"dim": dim, "time": time[i], "is_coupled": is_coupled[i]}

    print(df.head())
    sns.lineplot(data=df, x="dim", y="time", estimator="mean", ci="sd")
    plt.show()
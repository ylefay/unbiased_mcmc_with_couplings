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


def simulation_unbiased_generic(chain_key, x0, y0):
    proposal_cov = jnp.eye(dim) * 0.1
    q_hat = partial(random_walk_mh_proposal, chol_sigma=proposal_cov)

    def log_q(xp, x):
        return normal_logpdf(x=xp, mu=x, chol_sigma=proposal_cov)

    return unbiased_monte_carlo_estimation(chain_key, h, x0, y0, q_hat, log_q, log_target, lag, k, m)


@jax.vmap
def simulation_unbiased_target(key):
    chain_key, x0_key, y0_key = jax.random.split(key, 3)
    x0 = mu + chol_sigma @ jax.random.normal(x0_key, shape=(dim,))
    y0 = mu + chol_sigma @ jax.random.normal(y0_key, shape=(dim,))

    return simulation_unbiased_generic(chain_key, x0, y0)

@jax.vmap
def simulation_unbiased_offset(key):
    chain_key, x0_key, y0_key = jax.random.split(key, 3)
    x0 = mu + 1 + jax.random.normal(x0_key, shape=(dim,))
    y0 = mu + 1 + jax.random.normal(y0_key, shape=(dim,))

    return simulation_unbiased_generic(chain_key, x0, y0)

if __name__ == "__main__":
    dims = [1, 2, 3, 4, 5]
    n_samples = 10

    k = 100
    m = 10*k
    lag = 1

    df = pd.DataFrame(columns=["dim", "time", "initialization"])


    def log_target_builder(mu, chol_sigma):
        def log_target(x):
            return multivariate_normal.logpdf(x, mean=mu, cov=chol_sigma)
        return log_target
    
    def h(x):
        return 1.
    

    OP_key = jax.random.PRNGKey(randint(0, 1 << 30))
    key_dims = jax.random.split(OP_key, len(dims))

    for dim, key in tqdm(list(zip(dims, key_dims))):
        key_target, key_offset = jax.random.split(key, 2)
        keys = jax.random.split(key_target, 2*n_samples)
        chol_sigma = jnp.eye(dim) 
        mu = jnp.zeros(dim)
        log_target = log_target_builder(mu, chol_sigma)
        samples_unbiased, is_coupled, time = simulation_unbiased_target(keys)
        for i in range(n_samples):
            df.loc[len(df)] = {"dim": dim, "time": time[i], "initialization": "target"}

        samples_unbiased, is_coupled, time = simulation_unbiased_offset(keys)
        for i in range(n_samples):
            df.loc[len(df)] = {"dim": dim, "time": time[i], "initialization": "offset"}



    df["dim"] = df["dim"].astype(int)
    df["time"] = df["time"].astype(int)
    print(df.head())
    print(df.columns)
    sns.lineplot(data=df, x="dim", y="time", estimator="mean", errorbar="ci", hue="initialization")
    plt.xlabel("Dimension")
    plt.ylabel("Average meeting time")
    plt.legend()

    plt.show()
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
    chain_key, x0_key = jax.random.split(key, 2)
    x0 = jax.random.uniform(x0_key, shape=(dim,))

    chol_sigma = jnp.eye(dim) * 0.1
    q_hat = partial(random_walk_mh_proposal, chol_sigma=chol_sigma)

    def log_q(xp, x):
        return normal_logpdf(x=xp, mu=x, chol_sigma=chol_sigma)

    return default_monte_carlo_estimator(chain_key, h, x0, q_hat, log_q, log_target, n_chain)


@jax.vmap
def simulation_unbiased(key):

    chain_key, x0_key, y0_key = jax.random.split(key, 3)

    x0 = jax.random.uniform(x0_key, shape=(dim,))
    y0 = jax.random.uniform(y0_key, shape=(dim,))

    chol_sigma = jnp.eye(dim) * 0.1
    q_hat = partial(random_walk_mh_proposal, chol_sigma=chol_sigma)

    def log_q(xp, x):
        return normal_logpdf(x=xp, mu=x, chol_sigma=chol_sigma)

    return unbiased_monte_carlo_estimation(chain_key, h, x0, y0, q_hat, log_q, log_target, lag, k, m)


if __name__ == "__main__":

    dim = 1
    n_samples = 100

    k = 100
    m = 10*k
    lag = 1

    def log_target(x):
        return normal_logpdf(x=x, mu=jnp.ones(1), chol_sigma=jnp.eye(1))
    
    def h(x):
        return x[0]

    OP_key = jax.random.PRNGKey(randint(0, 1 << 30))
    keys = jax.random.split(OP_key, n_samples)

    samples_default = simulation_default(keys)
    print("simulation default done!")

    samples_unbiased, is_coupled, time = simulation_unbiased(keys)
    print("simulation unbiased done!")
    print(f"is coupled: {is_coupled}, time: {time}")

    import seaborn as sns
    print(samples_default.mean(), samples_unbiased.mean())
    print(f"Mean of the samples: \nDefault: {samples_default.mean()}\nUnbiased: {samples_unbiased.mean()}")
    print(f"Variance of the samples: \nDefault: {samples_default.var()}\nUnbiased: {samples_unbiased.var()}")
    sns.histplot(samples_default, kde=True)
    plt.title("Default Monte Carlo Estimator")
    plt.figure()
    sns.histplot(samples_unbiased, kde=True)
    plt.title("Unbiased Monte Carlo Estimator")
    plt.show()
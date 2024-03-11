import pickle
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal

from pymcmc_unbiased.maximal_coupling import coupling
from pymcmc_unbiased.monte_carlo_estimators import unbiased_monte_carlo_estimation

OP_key = jax.random.PRNGKey(0)

"""
Same as exp_coupling_time_wrt_to_eta_bis.py but with a bimodal normal distribution
"""
def pi_0(key):
    """
    Initial distribution
    """
    return jax.random.multivariate_normal(key, mean=4 * jnp.ones(dim, ), cov=1 * jnp.eye(dim, ))


def h(x):
    return 1.0


def log_target_builder(p, mu, chol_sigma):
    """
    Bimodal normal distribution:
        pN(m, s^2) + (1-p)N(-m, s^2)
    """

    def log_target(x):
        return jnp.log(p * multivariate_normal.pdf(x, mean=mu, cov=chol_sigma @ chol_sigma.T) +
                       (1 - p) * multivariate_normal.pdf(x, mean=-mu, cov=chol_sigma @ chol_sigma.T))

    return log_target


def random_walk_mh_proposal(key, x, chol_sigma):
    return x + chol_sigma @ jax.random.normal(key, (x.shape[-1],))


def simulation_unbiased_generic(chain_key, x0, y0, eta):
    chol_proposal = jnp.eye(dim) * 3.0
    cov_proposal = chol_proposal @ chol_proposal.T

    q_hat = partial(random_walk_mh_proposal, chol_sigma=chol_proposal)

    def log_q(xp, x):
        return multivariate_normal.logpdf(x=xp, mean=x, cov=cov_proposal)

    log_target = log_target_builder(0.5, jnp.ones(dim) * 4.0, jnp.eye(dim) * 1.0)

    return unbiased_monte_carlo_estimation(
        chain_key, h, x0, y0, q_hat, log_q, log_target, lag, k, m, max_iter=jnp.inf,
        coupling=partial(coupling, eta=eta)
    )


########### Two simulations with different initializations
@partial(jax.jit, static_argnums=(1,))
def simulation_unbiased_target(key, eta):
    chain_key, x0_key, y0_key = jax.random.split(key, 3)
    x0 = pi_0(x0_key)
    y0 = pi_0(y0_key)
    return simulation_unbiased_generic(chain_key, x0, y0, eta)


if __name__ == "__main__":
    result_target = dict()
    etas = [k / 100 for k in range(1, 100)]
    n_samples = 1000

    k = 10
    m = 10 * k
    lag = 1
    dim = 1

    key_etas = jax.random.split(OP_key, len(etas))

    # Compute the meeting time for each dimension and each initialization
    for i, eta in enumerate(etas):
        print(eta)
        keys = jax.random.split(key_etas[i], n_samples)

        samples_unbiased, is_coupled, time, meeting_time = jax.vmap(simulation_unbiased_target, in_axes=(0, None))(
            keys, eta
        )
        result_target[eta] = [samples_unbiased, is_coupled, time, meeting_time]

    with open("./results/results_coupling_time_target_wrt_to_eta_bis.pkl", "wb") as handle:
        pickle.dump(result_target, handle, protocol=pickle.HIGHEST_PROTOCOL)

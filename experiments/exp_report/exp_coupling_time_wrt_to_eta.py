import pickle
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal

from pymcmc_unbiased.maximal_coupling import coupling
from pymcmc_unbiased.monte_carlo_estimators import unbiased_monte_carlo_estimation
from utils import sample_invwishart

OP_key = jax.random.PRNGKey(0)


def h(x):
    return 1.0


def random_walk_mh_proposal(key, x, chol_sigma):
    return x + chol_sigma @ jax.random.normal(key, (x.shape[-1],))


def simulation_unbiased_generic(chain_key, x0, y0, chol_target, eta):
    chol_proposal = chol_target / jnp.sqrt(dim)
    cov_proposal = chol_proposal @ chol_proposal.T
    cov_target = chol_target @ chol_target.T

    q_hat = partial(random_walk_mh_proposal, chol_sigma=chol_proposal)

    def log_q(xp, x):
        return multivariate_normal.logpdf(x=xp, mean=x, cov=cov_proposal)

    def log_target(x):
        return multivariate_normal.logpdf(x, mean=jnp.zeros(dim, ), cov=cov_target)


    return unbiased_monte_carlo_estimation(
        chain_key, h, x0, y0, q_hat, log_q, log_target, lag, k, m, max_iter=jnp.inf,
        coupling=partial(coupling, eta=eta)
    )


########### Two simulations with different initializations
@partial(jax.jit, static_argnums=(1,))
def simulation_unbiased_target(key, eta):
    cov_target = sample_invwishart(key, df=dim, scale=jnp.eye(dim))
    chain_key, x0_key, y0_key = jax.random.split(key, 3)
    chol_target = jnp.linalg.cholesky(cov_target)
    x0 = chol_target @ jax.random.normal(x0_key, shape=(dim,))
    y0 = chol_target @ jax.random.normal(y0_key, shape=(dim,))
    return simulation_unbiased_generic(chain_key, x0, y0, chol_target, eta)


if __name__ == "__main__":
    result_target = dict()
    etas = [k/100 for k in range(1, 100)]
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

    with open("./results/results_coupling_time_target_wrt_to_eta.pkl", "wb") as handle:
        pickle.dump(result_target, handle, protocol=pickle.HIGHEST_PROTOCOL)

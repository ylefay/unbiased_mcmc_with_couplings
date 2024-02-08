import jax
import jax.numpy as jnp
from src.pymcmc_unbiased.maximal_coupling import maximal_coupling


def metropolis_hasting_coupling(keys, x0, y0, p_hat, q_hat, log_p, log_q):
    """
    Metropolis-Hastings coupling sampling procedure for a distribution p and q.
    :param keys: multiple jax.random.PRNGKey
    :param p_hat: callable
        sample from the marginal p
    :param q_hat: callable
        sample from the marginal q
    :param log_p: callable
        log density of p
    :param log_q: callable
        log density of q
    :return: jnp.ndarray,
        a sample (X, Y)
    """

    def iter_mh(carry, inp):
        key_k = inp
        coupling_key, sample_key = jax.random.split(key_k)
        x, y = carry
        _, couple_prop = maximal_coupling(coupling_key, ...)
        x_prop, y_prop = couple_prop
        U = jax.random.uniform(sample_key)
        accept_X = jnp.log(U) <= jnp.min(0, log_pi(x_prop) + log_q(x_prop, x) - log_pi(x) - log_q(x, x_prop))
        accept_Y = jnp.log(U) <= jnp.min(0, log_pi(y_prop) + log_q(y_prop, y) - log_pi(y) - log_q(y, y_prop))
        x = accept_X * x_prop + (1 - accept_X) * x
        y = accept_Y * y_prop + (1 - accept_Y) * y
        return (x, y), (x, y)

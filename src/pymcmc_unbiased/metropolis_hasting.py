import jax
import jax.numpy as jnp
from pymcmc_unbiased.maximal_coupling import maximal_coupling
from functools import partial


def mh_coupling(keys, coupling, x0, y0, q_hat, log_q, log_target):
    """
    Metropolis-Hastings coupling sampling procedure for a distribution p and q,
    Using maximal-coupling procedure.
    :param keys: multiple jax.random.PRNGKey
    :param q_hat: callable
        sample from the joint distribution q
    :param log_q: callable
        log density of q
    :return: jnp.ndarray,
        a sample (X, Y)
    """

    def iter_mh(carry, key_k):
        values, tau = carry
        x, y = values

        coupling_key, sample_key = jax.random.split(key_k, 2)
        n_iter_coupling, couple_prop = coupling(
            coupling_key, 
            p_hat=partial(q_hat, x=x), 
            q_hat=partial(q_hat, x=y),
            log_p=partial(log_q, x=x), 
            log_q=partial(log_q, x=y)
        )
        x_prop, y_prop = couple_prop

        log_U = -jax.random.exponential(sample_key)
        accept_X = log_U <= log_target(x_prop) + log_q(x_prop, x) - log_target(x) - log_q(x, x_prop)
        accept_Y = log_U <= log_target(y_prop) + log_q(y_prop, y) - log_target(y) - log_q(y, y_prop)

        x_next = jax.lax.select(accept_X, x_prop, x)
        y_next = jax.lax.select(accept_Y, y_prop, y)

        tau_next = tau + jax.lax.cond(
            (n_iter_coupling == 0) & accept_X & accept_Y, 
            lambda _: 0, 
            lambda _: 1, 
            None
        )
        return ((x_next, y_next), tau_next), (x_next, y_next)


    temp, (Xs, Ys) = jax.lax.scan(iter_mh, ((x0, y0), 0), keys)
    Xs = jnp.insert(Xs, 0, x0, axis=0)
    Ys = jnp.insert(Ys, 0, y0, axis=0)
    _, tau = temp

    return (Xs, Ys), tau


def run_chain(keys, x0, q_hat, log_q, log_target):
    def sample_from_transition_kernel(x, key_k):
        sample_key, accept_key = jax.random.split(key_k, 2)

        log_U = -jax.random.exponential(accept_key)
        x_prop = q_hat(sample_key, x=x)

        accept_X = log_U <= log_q(x_prop, x) + log_target(x_prop) - log_target(x) - log_q(x, x_prop)

        x_next = jax.lax.select(accept_X, x_prop, x)
        return x_next, x_next

    _, Xs = jax.lax.scan(sample_from_transition_kernel, x0, keys)
    Xs = jnp.insert(Xs, 0, x0, axis=0)
    return Xs

def mh_coupling_with_lag(keys, coupling, x0, y0, q_hat, log_q, log_target, lag=1):
    """
    Metropolis-Hastings coupling sampling procedure for a distribution p and q,
    Using maximal-coupling procedure.
    Algorithm 1. in Unbiased Markov Chain Monte Carlo:
        what, why and how
    Yves F. Atchadé, Pierre E. Jacob
    """
    keys_before_lag = keys.at[:lag].get()
    keys = keys.at[lag:].get()
    # First sample from the transition kernel lag times.
    Xs = run_chain(keys_before_lag, x0, q_hat, log_q, log_target)
    # Then using the coupled transition kernel
    chains, tau = mh_coupling(keys, coupling, Xs.at[-1].get(), y0, q_hat, log_q, log_target)
    tau += lag
    return Xs, chains, tau


mh_maximal_coupling_with_lag = partial(mh_coupling_with_lag, coupling=maximal_coupling)

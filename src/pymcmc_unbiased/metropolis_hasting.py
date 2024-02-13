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

    def iter_mh(carry, inp):
        key_k = inp
        coupling_key, sample_key = jax.random.split(key_k)
        carry, tau = carry
        x, y = carry
        _, couple_prop = coupling(
            coupling_key, 
            p_hat=partial(q_hat, x=x), 
            q_hat=partial(q_hat, x=y),
            log_p=partial(log_q, x=x), 
            log_q=partial(log_q, x=y)
        )
        x_prop, y_prop = couple_prop
        log_U = jnp.log(jax.random.uniform(sample_key))
        accept_X = log_U <= jnp.min(
            jnp.array([0, log_target(x_prop) + log_q(x_prop, x) - log_target(x) - log_q(x, x_prop)]))
        accept_Y = log_U <= jnp.min(
            jnp.array([0, log_target(y_prop) + log_q(y_prop, y) - log_target(y) - log_q(y, y_prop)]))
        tau += jax.lax.cond(jnp.any(x != y), lambda _: 1,
                            lambda _: 0, None)
        x = accept_X * x_prop + (1 - accept_X) * x
        y = accept_Y * y_prop + (1 - accept_Y) * y

        return ((x, y), tau), (x, y)

    _, chains = jax.lax.scan(iter_mh, ((x0, y0), 0), keys)
    Xs = jnp.insert(chains[0], 0, x0, axis=0)
    Ys = jnp.insert(chains[1], 0, y0, axis=0)
    _, tau = _
    return (Xs, Ys), tau


def run_chain(keys, x0, q_hat, log_q, log_target):
    def sample_from_transition_kernel(x, inp):
        sample_key_k = inp
        sample_key_uniform, sample_key_k = jax.random.split(sample_key_k, 2)
        U = jax.random.uniform(sample_key_uniform)
        x_prop = q_hat(sample_key_k, x=x)
        accept_X = jnp.log(U) <= jnp.min(
            jnp.array([0, log_q(x_prop, x) - log_q(x, x_prop) + log_target(x_prop) - log_target(x)]))
        x = accept_X * x_prop + (1 - accept_X) * x
        return x, x

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

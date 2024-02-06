import jax
import jax.numpy as jnp


def maximal_coupling(key, p_hat, q_hat, log_p, log_q):
    """
    Algorithm 2. in Pierre E. Jacob, John O'Leary, Yves F. Atchadé, 2019
    :param key: jax.random.PRNGKey
    :param p_hat: callable
        sample from the marginal p
    :param q_hat: callable
        sample from the marginal q
    :param log_p: callable
        log density of p
    :param log_q: callable
        log density of q
    :return: jnp.ndarray, int
        a sample (X, Y) from maximal coupling between p and q
        the number of trials before acceptance
    """

    def otherwise_fun(key):
        def iter_fun(inps):
            n_iter, _, _, keys = inps
            next_key, sample_key, accept_key = jax.random.split(key, 3)
            Y_star = q_hat(key)
            W_star = jax.random.uniform(accept_key, (1,)) * jnp.exp(log_q(Y_star))
            return n_iter + 1, W_star, Y_star, next_key

        def loop_condition(inps):
            _, W_star, Y_star = inps
            return W_star <= jnp.exp(log_p(Y_star))

        n_iter, _, Y_star_accepted = jax.lax.while_loop(loop_condition, iter_fun, (0, key))
        return n_iter, Y_star_accepted

    next_key, sample_key, accept_key = jax.random.split(key, 3)
    X = p_hat(sample_key)
    W = jax.random.uniform(accept_key, (1,)) * log_p(X)

    n_iter, couple = jax.lax.cond(W <= jnp.exp(log_q(X)), lambda _: (1, (X, X)), lambda _: otherwise_fun(sample_key),
                                  None)
    return n_iter, *couple

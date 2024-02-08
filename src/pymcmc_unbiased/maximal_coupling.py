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

    def otherwise_fun(op_key):
        def iter_fun(inps):
            n_iter, _, _, key = inps
            next_key, sample_key, accept_key = jax.random.split(key, 3)
            Y_star = q_hat(sample_key)
            log_W_star = jnp.log(jax.random.uniform(accept_key, (1,))) + log_q(Y_star)
            return n_iter + 1, log_W_star, Y_star, next_key

        def loop_condition(inps):
            _, log_W_star, Y_star = inps
            return log_W_star > log_p(Y_star)

        n_iter, _, Y_star_accepted = jax.lax.while_loop(loop_condition, iter_fun, (0, -jnp.inf, 0., op_key))
        return n_iter, Y_star_accepted

    next_key, sample_key, accept_key = jax.random.split(key, 3)
    X = p_hat(sample_key)
    W = jax.random.uniform(accept_key, (1,)) * log_p(X)
    coupled = jnp.log(W) <= log_q(X)
    n_iter, couple = jax.lax.cond(coupled, lambda _: (1, (X, X)), lambda _: otherwise_fun(sample_key),
                                  None)
    return n_iter, *couple

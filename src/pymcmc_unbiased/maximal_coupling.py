import jax
import jax.numpy as jnp


def maximal_coupling(key, p_hat, q_hat, log_p, log_q, eta=1):
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
    :param eta: float
        between 0 and 1, when eta is set to 1, the probability of coupling is maximal but the variance of the cost is infinite,
        if eta < 1, the variance of the cost.
    :return: jnp.ndarray, int
        a sample (X, Y) from maximal coupling between p and q
        the number of trials before acceptance
    """

    def otherwise_fun(op_key):
        def iter_fun(inps):
            n_iter, _, _, key = inps
            next_key, sample_key, accept_key = jax.random.split(key, 3)
            Y_star = q_hat(sample_key)
            log_W_star = jnp.log(jax.random.uniform(accept_key)) + log_q(Y_star)
            return n_iter + 1, log_W_star, Y_star, next_key

        def loop_condition(inps):
            _, log_W_star, Y_star, _ = inps
            return log_W_star <= log_p(Y_star) + jnp.log(eta)

        shape = q_hat(key).shape
        n_iter, _, Y_star_accepted, _ = jax.lax.while_loop(loop_condition, iter_fun,
                                                           (0, -jnp.inf, jnp.zeros(shape=shape), op_key))
        return n_iter, Y_star_accepted

    next_key, sample_key, accept_key = jax.random.split(key, 3)
    X = p_hat(sample_key)
    log_W = jnp.log(jax.random.uniform(accept_key))
    coupled = log_W <= jnp.min(jnp.array([jnp.log(eta), log_q(X) - log_p(X)]))
    n_iter, Y = jax.lax.cond(coupled, lambda _: (0, X), lambda _: otherwise_fun(sample_key),
                             None)
    return n_iter, (X, Y)

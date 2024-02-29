import jax
import jax.numpy as jnp


def maximal_coupling(key, p_hat, q_hat, log_p, log_q, eta=0.8, max_iter=10000):
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
        if eta < 1, the variance of the cost is bounded.
    :return: jnp.ndarray, int
        a sample (X, Y) from maximal coupling between p and q
        the number of trials before acceptance
    """

    log_eta = jnp.log(eta)

    def auxilary(aux_key, marginal):
        next_key, sample_key, accept_key = jax.random.split(aux_key, 3)

        # Sample X or Y*
        sample_var = marginal(sample_key)
        # Sample log_W or log_W*
        cond_var = -jax.random.exponential(accept_key)

        return next_key, sample_var, cond_var

    def otherwise_fun(fun_key):
        def iter_fun(inps):
            inps_key, _, _, n_iter = inps

            next_key, Y_star, log_W_star = auxilary(inps_key, q_hat)

            return next_key, Y_star, log_W_star, n_iter + 1

        def loop_condition(inps):
            _, Y_star, log_W_star, n_iter = inps
            return (log_W_star <= log_eta + log_p(Y_star) - log_q(Y_star)) & (n_iter < max_iter)

        next_key, Y_star, log_W_star = auxilary(fun_key, q_hat)

        _, Y_star_accepted, _, n_iter = jax.lax.while_loop(
            loop_condition,
            iter_fun,
            (next_key, Y_star, log_W_star, 1)
        )
        return n_iter, Y_star_accepted

    next_key, X, log_W = auxilary(key, p_hat)

    # condition 
    coupled = log_W <= jnp.minimum(log_eta, log_q(X) - log_p(X))
    # if
    n_iter, Y = jax.lax.cond(
        coupled,
        lambda: (0, X),  # true
        lambda: otherwise_fun(next_key)  # false
    )
    return n_iter, X, Y
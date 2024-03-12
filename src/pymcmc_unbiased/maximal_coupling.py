import jax
import jax.numpy as jnp


def modified_thorisson_coupling(key, p_hat, q_hat, log_p, log_q, phi, max_iter=10000):
    """
    This is the modified Thorisson's Algorithm described in the discussion by M. Gerber and A. Lee of the paper.
    :param key: jax.random.PRNGKey
    :param p_hat: callable
        sample from the marginal p
    :param q_hat: callable
        sample from the marginal q
    :param log_p: callable
        log density of p
    :param log_q: callable
        log density of q
    :param phi: callable
        Let w = q / p, phi_prime a function that takes values in [0, 1] s.t.phi_prime <= w.
        If we let phi_prime = min(eta, w), then we retrieve the coupling Algorithm 2. in Pierre E. Jacob, John O'Leary, Yves F. Atchadé, 2019.
        Here, we let phi_prime = min(phi, w) to ensure the previous pointwise inequality.
        Hence, setting phi = eta, we retrieve the coupling Algorithm 2.
    :return: jnp.ndarray, int
        a sample (X, Y) from a coupling between p and q
        the number of trials before acceptance
    """

    def log_phi_prime(x):
        """
        Ensuring that phi_prime is always less than or equal to w.
        """
        return jnp.minimum(jnp.log(phi(x)), log_q(x) - log_p(x))

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
            """
                W <= phi_prime(x) / w(x)
            <=> W <= min(phi(x), w(x)) / w(x)
            <=> W <= min(phi(x), q/p(x)) / (q/p(x))
            <=> W <= min(phi(x) * p(x) / q(x), 1)
            <=> W <= phi(x) * p(x)/q(x)

            With phi(x) = eta,
                W <= eta * p(x) / q(x)
            <=> W <= min(eta, w(x)) / w(x)
            :param inps:
            :return:
            """
            _, Y_star, log_W_star, n_iter = inps
            return (log_W_star <= log_phi_prime(Y_star) + log_p(Y_star) - log_q(Y_star)) & (n_iter < max_iter)

        next_key, Y_star, log_W_star = auxilary(fun_key, q_hat)

        _, Y_star_accepted, _, n_iter = jax.lax.while_loop(
            loop_condition,
            iter_fun,
            (next_key, Y_star, log_W_star, 1)
        )
        return n_iter, Y_star_accepted

    next_key, X, log_W = auxilary(key, p_hat)

    # condition
    coupled = log_W <= log_phi_prime(X)
    # if
    n_iter, Y = jax.lax.cond(
        coupled,
        lambda: (0, X),  # true
        lambda: otherwise_fun(next_key)  # false
    )
    return n_iter, X, Y


def coupling(key, p_hat, q_hat, log_p, log_q, eta=1.0, max_iter=1e4):
    """
    This is the Thorisson's Algorithm with an extra parameter eta.
    Algorithm 2. in Pierre E. Jacob, John O'Leary, Yves F. Atchadé, 2019.
    It is a maximal coupling when eta = 1.
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
        between 0 and 1, when eta is set to 1, the probability of coupling is maximal but the variance of the cost is infinite (can be?, there is only a lower bound),
        if eta < 1, the variance of the cost is bounded.
    :return: jnp.ndarray, int
        a sample (X, Y) from a coupling between p and q
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

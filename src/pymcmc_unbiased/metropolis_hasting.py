import jax


def mh_single_kernel(key, x, q_hat, log_q, log_target):
    sample_key, accept_key = jax.random.split(key, 2)

    x_prop = q_hat(sample_key, x)
    log_U = -jax.random.exponential(accept_key)

    accept = log_U <= log_target(x_prop) + log_q(x_prop, x) - log_target(x) - log_q(x, x_prop)

    x_next = jax.lax.cond(
        accept,
        lambda: x_prop,
        lambda: x
    )

    return x_next


def mh_coupled_kernel(key, x, y, coupling, q_hat, log_q, log_target):
    sample_key, accept_key = jax.random.split(key, 2)

    n_iter_coupling, x_prop, y_prop = coupling(  # c'est un peu plus clair comme ça
        key=sample_key,
        p_hat=lambda lmbd_key: q_hat(lmbd_key, x),
        q_hat=lambda lmbd_key: q_hat(lmbd_key, y),
        log_p=lambda z: log_q(x, z),
        log_q=lambda z: log_q(y, z)
    )

    log_U = -jax.random.exponential(accept_key)
    accept_X = log_U <= log_target(x_prop) + log_q(x_prop, x) - log_target(x) - log_q(x, x_prop)
    accept_Y = log_U <= log_target(y_prop) + log_q(y_prop, y) - log_target(y) - log_q(y, y_prop)

    x_next = jax.lax.cond(
        accept_X,
        lambda: x_prop,
        lambda: x
    )
    y_next = jax.lax.cond(
        accept_Y,
        lambda: y_prop,
        lambda: y
    )
    coupled = (n_iter_coupling == 0) & accept_X & accept_Y

    return coupled, x_next, y_next

import jax
import jax.numpy as jnp

from pymcmc_unbiased.maximal_coupling import coupling
from pymcmc_unbiased.metropolis_hasting import mh_single_kernel, mh_coupled_kernel


def default_monte_carlo_estimator(key, h, x0, q_hat, log_q, log_target, n_chain, burnin_period=0):
    zero_like_h = jnp.zeros_like(h(x0))

    def body_loop(time, val):
        key, x, curr_sum = val
        next_key, sample_key = jax.random.split(key, 2)

        curr_sum += jax.lax.cond(time >= burnin_period, lambda: h(x), lambda: zero_like_h)
        x_next = mh_single_kernel(
            key=sample_key,
            x=x,
            q_hat=q_hat,
            log_q=log_q,
            log_target=log_target
        )

        return next_key, x_next, curr_sum

    _, _, curr_sum = jax.lax.fori_loop(
        0,
        n_chain + burnin_period,
        body_loop,
        (key, x0, zero_like_h)
    )

    return curr_sum / n_chain


def unbiased_monte_carlo_estimation(key, h, x0, y0, q_hat, log_q, log_target, lag, k, m, max_iter=1e6, coupling=coupling):
    meeting_time = max_iter

    def v(t):
        return jnp.floor((t - k) / lag) - jnp.ceil(jnp.maximum(lag, t - m) / lag) + 1.

    # case k == 0
    mcmc = jax.lax.cond(
        k == 0,
        lambda: h(x0),
        lambda: 0.
    )

    def body_loop_1(time, val):
        key, x_prev, mcmc = val
        next_key, sample_key = jax.random.split(key, 2)

        x_next = mh_single_kernel(
            key=sample_key,
            x=x_prev,
            q_hat=q_hat,
            log_q=log_q,
            log_target=log_target
        )

        mcmc = jax.lax.cond(
            time < k,
            lambda: mcmc,
            lambda: mcmc + h(x_next)
        )

        return next_key, x_next, mcmc

    next_key, x_next, mcmc = jax.lax.fori_loop(
        1,
        lag + 1,
        body_loop_1,
        (key, x0, mcmc)
    )

    # case k == 0
    bias_cancel = jax.lax.cond(
        k == 0,
        lambda: v(lag) * (h(x_next) - h(y0)),
        lambda: 0.
    )

    def body_loop_2(inps):
        key, x_prev, y_prev, mcmc, bias_cancel, is_coupled, time, meeting_time = inps
        next_key, sample_key = jax.random.split(key, 2)

        def coupled_case():
            x_next = mh_single_kernel(
                key=sample_key,
                x=x_prev,
                q_hat=q_hat,
                log_q=log_q,
                log_target=log_target
            )
            return x_next, x_next, is_coupled

        def uncoupled_case():
            is_coupled, x_next, y_next = mh_coupled_kernel(
                key=sample_key,
                x=x_prev,
                y=y_prev,
                coupling=coupling,
                q_hat=q_hat,
                log_q=log_q,
                log_target=log_target
            )
            return x_next, y_next, is_coupled

        x_next, y_next, is_coupled = jax.lax.cond(
            is_coupled,
            coupled_case,
            uncoupled_case
        )

        mcmc = jax.lax.cond(
            (time < k) | (time > m),
            lambda: mcmc,
            lambda: mcmc + h(x_next)
        )

        bias_cancel = jax.lax.cond(
            (time < k + lag) | is_coupled,
            lambda: bias_cancel,
            lambda: bias_cancel + v(time) * (h(x_next) - h(y_next))
        )

        meeting_time = jax.lax.cond(
            is_coupled,
            lambda: jnp.minimum(meeting_time, time),
            lambda: meeting_time
        )

        return next_key, x_next, y_next, mcmc, bias_cancel, is_coupled, time + 1, meeting_time

    def cond_loop_2(inps):
        _, _, _, _, _, is_coupled, time, _ = inps
        return (~is_coupled | (time <= m)) & (time < max_iter)

    _, _, _, mcmc, bias_cancel, is_coupled, time, meeting_time = jax.lax.while_loop(
        cond_loop_2,
        body_loop_2,
        (next_key, x_next, y0, mcmc, bias_cancel, False, lag + 1, max_iter)
    )

    return (mcmc + bias_cancel) / (m - k + 1), is_coupled, time, meeting_time

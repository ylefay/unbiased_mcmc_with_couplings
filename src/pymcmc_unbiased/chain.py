import jax
import jax.numpy as jnp

from pymcmc_unbiased.maximal_coupling import maximal_coupling
from pymcmc_unbiased.metropolis_hasting import mh_single_kernel, mh_coupled_kernel


def run_chain_coupled(key, x0, y0, q_hat, log_q, log_target, lag, dim, n_chain):
    """
        This function is just if you want to plot chains, 
        should not be used in other functions, 
        slow.

        Do not vectorize this.
    """

    # ---------------------------------------------INITIALISATION------------------------------------------
    Xs = jnp.empty((n_chain, dim))
    Ys = jnp.empty((n_chain - lag, dim))

    Xs = Xs.at[0].set(x0)
    Ys = Ys.at[0].set(y0)

    # ----------------------------------------------START X-----------------------------------------
    def body_loop_1(i, val):
        key, Xs = val
        next_key, sample_key = jax.random.split(key, 2)

        Xs = Xs.at[i].set(mh_single_kernel(
            key=sample_key, 
            x=Xs[i - 1], 
            q_hat=q_hat, 
            log_q=log_q, 
            log_target=log_target
        ))
        return next_key, Xs

    next_key, Xs = jax.lax.fori_loop(
        1,
        lag + 1,
        body_loop_1,
        (key, Xs)
    )
        
    
    # ----------------------------------------------Sample X and Y-----------------------------------------
    
    def body_loop_2(i, val):
        key, Xs, Ys, is_coupled, time_coupled = val
        next_key, sample_key = jax.random.split(key, 2)

        def coupled_case(time_coupled):
            x_next = mh_single_kernel(
                key=sample_key,
                x=Xs[i - 1],
                q_hat=q_hat,
                log_q=log_q,
                log_target=log_target
            )
            return x_next, x_next, is_coupled, time_coupled

        def uncoupled_case(time_coupled):
            is_coupled, x_next, y_next = mh_coupled_kernel(
                key=sample_key,
                x=Xs[i - 1],
                y=Ys[i - lag - 1],
                coupling=maximal_coupling,
                q_hat=q_hat,
                log_q=log_q,
                log_target=log_target
            )
            time_coupled = jax.lax.cond(is_coupled, lambda: i, lambda: time_coupled)
            return x_next, y_next, is_coupled, time_coupled

        
        x_next, y_next, is_coupled, time_coupled = jax.lax.cond(
            is_coupled,
            coupled_case,
            uncoupled_case,
            operand=time_coupled
        )

        Xs = Xs.at[i].set(x_next)
        Ys = Ys.at[i - lag].set(y_next)

        return next_key, Xs, Ys, is_coupled, time_coupled

    _, Xs, Ys, is_coupled, time_coupled = jax.lax.fori_loop(
        lag + 1, 
        n_chain, 
        body_loop_2, 
        (next_key, Xs, Ys, False, n_chain)
    )

    return is_coupled, time_coupled, Xs, Ys
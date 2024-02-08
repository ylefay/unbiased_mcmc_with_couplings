from src.pymcmc_unbiased.mh import metropolis_hasting_maximal_coupling_with_lag


def wrapped_maximal_coupling_mh(q_hat, log_q, log_target, lag):
    def wrapped(keys, x0, y0):
        return metropolis_hasting_maximal_coupling_with_lag(keys, x0, y0, q_hat, log_q, log_target, lag)

    return wrapped

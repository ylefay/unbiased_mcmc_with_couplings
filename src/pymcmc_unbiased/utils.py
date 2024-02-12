import numpy as np


def reconstruct_chains(Xs_before_lag, chains):
    """
    Reconstruct the full chains from the output of mh_coupling_with_lag.
    :param Xs_before_lag: jnp.ndarray
        the Xs before the lag, comprising the initial value
    :param chains: tuple (2, ) of jnp.ndarray
        the chains, with X having the last value before lag (X_l), and Y having the first value (init), Y_0
    :return: tuple (2, ) of jnp.ndarray
        the reconstructed chains ((X_0, ... X_lag, X_{lag+1}, ..., X_T), (Y_0, ..., Y_{T-lag}))
    """
    Xs, Ys = chains
    Xs = np.concatenate([Xs_before_lag, Xs[1:]], axis=0)
    return Xs, Ys


def get_coupling_time(chains):
    """
    Get the coupled time tau given reconstructed chains, with
    tau = inf{t >= L : X_t = Y_{t-L}}
    i.e., first t such that chains[0][t]==chains[1][t-lag]
    """
    Xs, Ys = chains
    lag = len(Xs) - len(Ys)
    Xs = Xs[lag:]
    tau = np.argmin(np.linalg.norm(Xs - Ys, axis=-1)) + lag
    return tau
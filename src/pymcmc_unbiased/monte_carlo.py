import jax
import jax.numpy as jnp


def unbiased(test_function, k, l, lag, chains):
    """
        Unbiased estimator of the expectation of a test function.
        :param test_function: callable
            the test function, h
        :param chains: tuple (2, ) of jnp.ndarray
            the chains (with X having the first L values)
        :return: float
            the unbiased estimator, Eq. 1.2.7 from Ref2.
    """

    h = test_function
    Xs, Ys = chains
    tau = 100  # to fix
    hXs = jax.vmap(h)(Xs)
    Ys = Ys[:tau]
    hYs = jax.vmap(h)(Ys)
    weight = 1 / (l - k + 1)
    MCMC = weight * jnp.sum(hXs.at[k:l + 1].get())
    diff = hXs.at[k + lag:tau + lag].get() - hYs[k:tau].get()

    @jax.vmap
    def fun_weight(t):
        return (jnp.floor((t - k) / lag) - jnp.ceil(jnp.max([lag, t - l]) / lag) + 1) / (l - k + 1)

    weights = fun_weight(jnp.arange(k + lag, tau - 1))
    bias_correction = weights @ diff
    Hkl = MCMC + bias_correction
    return Hkl

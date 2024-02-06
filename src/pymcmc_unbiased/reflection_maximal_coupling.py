import jax
import jax.numpy as jnp


def reflection_maximal_coupling(key, spherical_distribution_hat, mus, chol_sigma):
    """
    Reflection-maximal coupling sampling procedure for a spherical distribution s.
    :param key: jax.random.PRNGKey
    :param spherical_distribution_hat: callable
        sample from the spherical distribution
    :return: jnp.ndarray,
        a sample (X, Y)
    """
    s = spherical_distribution_hat
    mu1, mu2 = mus
    z = jnp.linalg.inv(chol_sigma) @ (mu1 - mu2)
    e = z / jnp.linalg.norm(z)
    next_key, sample_key, accept_key = jax.random.split(key, 3)
    X_dot = s(sample_key)
    U = jax.random.uniform(accept_key)
    Y_dot = jax.lax.cond(U <= jnp.min(1, s(X_dot + z) / s(X_dot)), lambda _: X_dot + z,
                         lambda _: X_dot - 2 * (e.T @ X_dot) * e)
    X, Y = mu1 + chol_sigma @ X_dot, mu2 + chol_sigma @ Y_dot
    return X, Y

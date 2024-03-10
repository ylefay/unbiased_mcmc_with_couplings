import jax
import jax.numpy as jnp


def sample_invwishart(key, df, scale):
    """
    Sample from the inverse Wishart distribution.
    """
    p = scale.shape[0]
    assert scale.shape[1] == p
    chol = jnp.linalg.cholesky(scale)
    z = chol @ jax.random.normal(key, shape=(df, p))
    return jnp.linalg.inv(z.T @ z)

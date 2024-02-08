import jax.numpy as jnp


def normal_logpdf(x, mu, chol_sigma):
    _, log_det = jnp.linalg.slogdet(chol_sigma)
    return -0.5 * (jnp.log(2 * jnp.pi)) - log_det + (
            (x - mu).T @ jnp.linalg.inv(chol_sigma) @ (x - mu))

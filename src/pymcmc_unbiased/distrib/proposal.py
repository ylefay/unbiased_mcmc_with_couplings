import jax.random


def random_walk_mh_proposal(chol_sigma, key, x):
    return x + chol_sigma @ jax.random.normal(key, (x.shape[0],))

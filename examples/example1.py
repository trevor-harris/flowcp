import flowcp
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# generate fake data
key = jax.random.key(0)
key, key_cal, key_hat, key_test = jax.random.split(key, 4)

ycal = jax.random.normal(key_cal, (500, 32))
ycal_hat = jax.random.normal(key_hat, (500, 32))
ytest_hat = jax.random.normal(key_test, (1, 32))

# find tau's for all alpha range
alpha = jnp.linspace(0.0, 1.0, ycal.shape[0])
l2_score = flowcp.score.l2_nonconf()
l2_score.fit(ycal, ycal_hat, alpha)

# sample at 20 steps, init from ycal
l2_samp = flowcp.flow(l2_score, ycal, ytest_hat, 20)
print(l2_samp.shape)

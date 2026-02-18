import flowcp
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

#### generate "data" and "predictions"
key = jax.random.key(0)
key, key_cal, key_hat, key_test = jax.random.split(key, 4)

ycal = jax.random.normal(key_cal, (500, 32, 32, 3))
ycal_hat = jax.random.normal(key_hat, (500, 32, 32, 3))
ytest_hat = jax.random.normal(key_test, (1, 32, 32, 3))

##### define score
sob_score = flowcp.score.sobolev_nonconf(lam = 10.0)

#### sample at a fixed alpha level
alpha = 0.1
sob_score.fit(ycal, ycal_hat, alpha)

# sample
sob_samp = flowcp.flow(sob_score, ycal, ytest_hat, 20)

# verify score error
print(sob_score.score(sob_samp, ytest_hat) - sob_score.tau)

#### sample across a range of alphas
alpha = jnp.linspace(0.0, 1.0, ycal.shape[0])
sob_score.fit(ycal, ycal_hat, alpha)

# sample
sob_samp = flowcp.flow(sob_score, ycal, ytest_hat, 20)

# verify score error
print(sob_score.score(sob_samp, ytest_hat) - sob_score.tau)



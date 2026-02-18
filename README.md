# FlowCP

## Overview
`flowcp` implements Flow Based Conformal Predictive Distributions. 

## Install
```bash
git clone https://github.com/trevor-harris/flowcp
pip install flowcp/
```

## Examples

### Example 1 - White noise predictor (N, p)
```python
import flowcp
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

#### generate "data" and "predictions"
key = jax.random.key(0)
key, key_cal, key_hat, key_test = jax.random.split(key, 4)

ycal = jax.random.normal(key_cal, (500, 32))
ycal_hat = jax.random.normal(key_hat, (500, 32))
ytest_hat = jax.random.normal(key_test, (1, 32))

##### define score
l2_score = flowcp.score.l2_nonconf()

#### sample at a fixed alpha level
alpha = 0.1
l2_score.fit(ycal, ycal_hat, alpha)

# sample
l2_samp = flowcp.flow(l2_score, ycal, ytest_hat, 20)

# verify score error
print(jnp.max(l2_score.score(l2_samp, ytest_hat) - l2_score.tau))

#### sample across a range of alphas
alpha = jnp.linspace(0.0, 1.0, ycal.shape[0])
l2_score.fit(ycal, ycal_hat, alpha)

# sample
l2_samp = flowcp.flow(l2_score, ycal, ytest_hat, 20)

# verify score error
print(jnp.max(l2_score.score(l2_samp, ytest_hat) - l2_score.tau))
```

### Example 2 - White noise predictor (N, h, w, c)
```python
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
print(jnp.max(sob_score.score(sob_samp, ytest_hat) - sob_score.tau))

#### sample across a range of alphas
alpha = jnp.linspace(0.0, 1.0, ycal.shape[0])
sob_score.fit(ycal, ycal_hat, alpha)

# sample
sob_samp = flowcp.flow(sob_score, ycal, ytest_hat, 20)

# verify score error
print(jnp.max(sob_score.score(sob_samp, ytest_hat) - sob_score.tau))
```
## Notes
This package is under active development.

## Cite us

If you use `flowcp` in an academic paper, please cite [1]

```bibtex
@article{harris2026flow,
  title={Flow-Based Conformal Predictive Distributions},
  author={Harris, Trevor},
  journal={arXiv preprint arXiv:2602.07633},
  year={2026}
}
```
## References
<a id='1'>[1]</a>
Harris T.; 
Flow-Based Conformal Predictive Distributions;
[arxiv link](https://arxiv.org/abs/2602.07633v1)

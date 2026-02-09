# FlowCP

## Overview
`flowcp` implements Flow Based Conformal Predictive Distributions. 

## Install
```bash
git clone https://github.com/trevor-harris/flowcp
pip install flowcp/
```

## Examples

```python
import jax 
import jax.numpy as jnp
import flowcp

# generate fake data
key = jax.random.key(0)
key, key_cal, key_hat, key_test = jax.random.split(key, 4)
ycal = jax.random.normal(key_cal, (500, 32))
ycal_hat = jax.random.normal(ycal (500, 32))
ytest = jax.random.normal(ycal (1, 32))

# set alpha level, find tau_alpha
alpha = 0.1
l2_score = flowcp.score.l2_nonconf()
l2_score.fit(ycal, ycal_hat, alpha)

# sample
l2_samp = integrate(l2_score, ycal, ytest_hat, 20)

```
## Notes
This package is under active development.

##
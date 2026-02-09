import jax
import jax.numpy as jnp

def velocity(score, y, yhat, tau, lam):
    s_val, s_grad = score.score_grad(y, yhat)
    s_diff = s_val - tau
    s_norm = jnp.sum(s_grad**2, axis = (1, 2, 3), keepdims=True)
    vel = -lam * s_diff[:,None,None,None] * (s_grad / s_norm)
    return vel
velocity = jax.jit(velocity, static_argnums=0)

def flow(score, y0, yhat, steps=20):
  y = y0

  # pre-compute lambda
  s_val, s_grad = score.score_grad(y, yhat)
  lam = jnp.max(jnp.log(jnp.abs(s_val)/1e-6))
  dt = 1/steps

  # select alpha levels
  n_cal = score.tau.shape[0]
  n_samp = y0.shape[0]
  tau_idx = jnp.linspace(0, n_cal, n_samp, dtype = int)
  tau = score.tau[tau_idx]

  for _ in range(steps):
    vel = velocity(score, y, yhat, tau, lam)
    y += dt*vel
  return y
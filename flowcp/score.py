import jax 
import jax.numpy as jnp

# ============================================================
# L2
# ============================================================
class l2_nonconf:
  def __init__(self):
    pass

  def fit(self, y: jnp.ndarray, yhat: jnp.ndarray, alpha: jnp.ndarray):
    """
    alpha:   scalar or (A,) array in [0,1]
    """
    self.alpha = alpha
    self.n = y.shape[0]
    self.k_alpha = jnp.ceil((1.0 - self.alpha) * (self.n + 1)).astype(int)
    self.cal_scores = self.batch_score(y, yhat)                 # (N,)
    self.tau = jnp.sort(self.cal_scores)[self.k_alpha]          # scalar or (A,)

  def _score(self, y: jnp.ndarray, yhat: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.mean((y - yhat) ** 2))
  
  def batch_score(self, y: jnp.ndarray, yhat: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(self._score, in_axes=(0, 0))(y, yhat)
  
  score      = jax.vmap(_score, (None, 0, None))
  score_grad = jax.vmap(jax.value_and_grad(_score, argnums=1), (None, 0, None))


# ============================================================
# L1
# ============================================================
class l1_nonconf():
  def __init__(self):
    pass

  def fit(self, y, yhat, alpha):
    self.alpha = alpha
    self.n = y.shape[0]
    self.k_alpha = jnp.ceil((1-self.alpha)*(self.n+1)).astype(int)
    scores = self.score(y, yhat)
    self.tau = jnp.sort(scores)[self.k_alpha]

  def _score(self, y, yhat):
    val = jnp.mean(jnp.abs(y - yhat))
    return val
  
  def batch_score(self, y: jnp.ndarray, yhat: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(self._score, in_axes=(0, 0))(y, yhat)

  score = jax.vmap(_score, (None, 0, 0))
  score_grad = jax.vmap(jax.value_and_grad(_score, argnums = 1), (None, 0, 0))

# ============================================================
# Huber
# ============================================================
class huber_nonconf():
  def __init__(self, delta = 1):
    self.delta = delta

  def fit(self, y, yhat, alpha):
    self.alpha = alpha
    self.n = y.shape[0]
    self.k_alpha = jnp.ceil((1-self.alpha)*(self.n+1)).astype(int)
    scores = self.score(y, yhat)
    self.tau = jnp.sort(scores)[self.k_alpha]

  def _score(self, y, yhat):
    resid = jnp.abs(yhat - y)
    quad = 0.5*resid**2
    lin  = self.delta * (resid - 0.5 * self.delta)
    val = jnp.mean(quad * (resid < self.delta) + lin * (resid >= self.delta))
    return val
  
  def batch_score(self, y: jnp.ndarray, yhat: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(self._score, in_axes=(0, 0))(y, yhat)

  score = jax.vmap(_score, (None, 0, 0))
  score_grad = jax.vmap(jax.value_and_grad(_score, argnums = 1), (None, 0, 0))

# ============================================================
# Gaussian NLL
# ============================================================
class gauss_nonconf():
  def __init__(self):
    pass

  def fit(self, y, yhat, alpha):
    self.alpha = alpha
    self.n = y.shape[0]
    self.k_alpha = jnp.ceil((1-self.alpha)*(self.n+1)).astype(int)

    resid = y - yhat
    self.mu = jnp.mean(resid, axis = 0)
    self.cov = jnp.std(resid, axis = 0)
    self.cov = jnp.diag(self.cov)

    scores = self.score(y, yhat)
    self.tau = jnp.sort(scores)[self.k_alpha]

  def _score(self, y, yhat):
    resid = y - yhat
    lpdf = jax.scipy.stats.multivariate_normal.logpdf
    return -lpdf(resid, self.mu, self.cov)

  def batch_score(self, y: jnp.ndarray, yhat: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(self._score, in_axes=(0, 0))(y, yhat)

  score = jax.vmap(_score, (None, 0, 0))
  score_grad = jax.vmap(jax.value_and_grad(_score, argnums = 1), (None, 0, 0))


class t_nonconf:
  def __init__(self, nu):
    self.nu = nu

  def fit(self, y, yhat, alpha):
    self.alpha = alpha
    self.n = y.shape[0]
    self.k_alpha = jnp.ceil((1-self.alpha)*(self.n+1)).astype(int)

    resid = y - yhat
    self.mu = jnp.mean(resid, axis = 0)
    # self.cov = jnp.cov(resid.T)
    # self.cov = self.cov + 1e-5 * jnp.eye(self.cov.shape[0])
    self.cov = jnp.std(resid, axis = 0)
    self.cov = jnp.diag(self.cov)

    scores = self.score(y, yhat)
    self.tau = jnp.sort(scores)[self.k_alpha]

  def _score(self, y, yhat):
    r = y - yhat - self.mu
    L = jnp.linalg.cholesky(self.cov)  # Sigma must be PD
    z = jax.scipy.linalg.solve_triangular(L, r, lower=True)
    quad = jnp.dot(z, z)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    d = y.shape[0]

    return (
        0.5 * logdet
        + 0.5 * (self.nu + d) * jnp.log1p(quad / self.nu)
        + jax.scipy.special.gammaln(self.nu / 2)
        - jax.scipy.special.gammaln((self.nu + d) / 2)
        + 0.5 * d * jnp.log(self.nu * jnp.pi)
    )
  
  def batch_score(self, y: jnp.ndarray, yhat: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(self._score, in_axes=(0, 0))(y, yhat)

  score = jax.vmap(_score, (None, 0, 0))
  score_grad = jax.vmap(jax.value_and_grad(_score, argnums = 1), (None, 0, 0))

# ============================================================
# Sobolev (gradient) score
# ============================================================
class sobolev_nonconf:
  def __init__(self, lam: float = 1.0):
    self.lam = float(lam)

  def fit(self, y: jnp.ndarray, yhat: jnp.ndarray, alpha: jnp.ndarray):
    self.alpha = alpha
    self.n = y.shape[0]
    self.k_alpha = jnp.ceil((1.0 - self.alpha) * (self.n + 1)).astype(int)
    self.cal_scores = self.batch_score(y, yhat)
    self.tau = jnp.sort(self.cal_scores)[self.k_alpha]

  def batch_score(self, y: jnp.ndarray, yhat: jnp.ndarray) -> jnp.ndarray:
    e = y - yhat
    gx = jnp.roll(e, -1, axis=2) - e  # along W (axis=2 for (N,H,W,C))
    gy = jnp.roll(e, -1, axis=1) - e  # along H
    e0 = jnp.mean(e * e, axis=range(1, y.ndim))
    ex = jnp.mean(gx * gx, axis=range(1, y.ndim))
    ey = jnp.mean(gy * gy, axis=range(1, y.ndim))
    return jnp.sqrt(e0 + self.lam * (ex + ey))

  def _score(self, y: jnp.ndarray, yhat: jnp.ndarray) -> jnp.ndarray:
    e = y - yhat  # (H,W,C)
    gx = jnp.roll(e, -1, axis=1) - e  # along W (axis=1 for (H,W,C))
    gy = jnp.roll(e, -1, axis=0) - e  # along H
    e0 = jnp.mean(e * e)
    ex = jnp.mean(gx * gx)
    ey = jnp.mean(gy * gy)
    return jnp.sqrt(e0 + self.lam * (ex + ey))

  score      = jax.vmap(_score, (None, 0, None))
  score_grad = jax.vmap(jax.value_and_grad(_score, argnums=1), (None, 0, None))
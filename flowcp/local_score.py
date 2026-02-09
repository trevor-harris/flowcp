import jax
import jax.numpy as jnp


# ------------------------
# Localization: x features
# ------------------------
def _avg_pool_to(u: jnp.ndarray, out_hw=(8, 8)) -> jnp.ndarray:
  """
  Simple deterministic pooling to a fixed grid using resize (linear).
  u: (H,W,C) -> (outH,outW,C)
  """
  outH, outW = out_hw
  # jax.image.resize is available in JAX; linear is smooth and stable.
  return jax.image.resize(u, (outH, outW, u.shape[-1]), method="linear")


def default_x_features(x: jnp.ndarray, out_hw=(8, 8), eps=1e-6) -> jnp.ndarray:
  """
  x: (H,W,Cx) -> feature vector (d,)
  Features = [per-channel mean, per-channel std, pooled grid values]
  """
  x = x.astype(jnp.float32)
  mu = jnp.mean(x, axis=(0, 1))
  sd = jnp.std(x, axis=(0, 1)) + eps
  xp = _avg_pool_to(x, out_hw=out_hw).reshape(-1)
  return jnp.concatenate([mu, sd, xp], axis=0)


def fit_phi_standardizer(Phi: jnp.ndarray, eps=1e-6):
  mu = jnp.mean(Phi, axis=0)
  sd = jnp.std(Phi, axis=0) + eps
  return mu, sd


def rbf_weights(Phi_cal: jnp.ndarray, phi_star: jnp.ndarray, mu: jnp.ndarray, sd: jnp.ndarray, h: float = 1.0):
  Z = (Phi_cal - mu) / sd
  z = (phi_star - mu) / sd
  d2 = jnp.sum((Z - z[None, :]) ** 2, axis=1)
  w = jnp.exp(-0.5 * d2 / (h ** 2))
  w = w / (jnp.sum(w) + 1e-12)
  return w


def weighted_quantile(values: jnp.ndarray, weights: jnp.ndarray, q: float):
  """
  values: (N,), weights: (N,) nonnegative, q in [0,1]
  Not intended to be jitted (argsort/searchsorted).
  """
  idx = jnp.argsort(values)
  v = values[idx]
  w = weights[idx]
  cw = jnp.cumsum(w)
  cw = cw / (cw[-1] + 1e-12)
  j = jnp.searchsorted(cw, q, side="left")
  j = jnp.clip(j, 0, v.shape[0] - 1)
  return v[j]


def _robust_scale(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
  med = jnp.median(x)
  mad = jnp.median(jnp.abs(x - med))
  return mad + eps


# ------------------------
# Spectral distance utility
# ------------------------
def _log_power_spectrum(u: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
  """
  u: (H,W) -> log power spectrum (Hr,Wr)
  We remove the mean and use log1p(power).
  """
  u = u - jnp.mean(u)
  F = jnp.fft.rfft2(u)
  P = jnp.abs(F) ** 2
  return jnp.log1p(P + eps)


def _logspec_dist(y: jnp.ndarray, yhat: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
  """
  y,yhat: (H,W,C) -> scalar distance
  Mean over channels of MSE in log-power-spectral domain.
  """
  y = y.astype(jnp.float32)
  yhat = yhat.astype(jnp.float32)

  C = y.shape[-1]

  def per_c(c):
    Sy = _log_power_spectrum(y[..., c], eps=eps)
    Sh = _log_power_spectrum(yhat[..., c], eps=eps)
    return jnp.mean((Sy - Sh) ** 2)

  return jnp.mean(jax.vmap(per_c)(jnp.arange(C)))

# -----------------------------------------
# Localized combined score: L2 + log-spectrum
# -----------------------------------------
class local_nonconf:
  """
  Score(y, yhat) = weighted RMS of:
    - l2:      sqrt(mean((y - yhat)^2))
    - logspec: sqrt(mean((log P(y) - log P(yhat))^2))

  Each term is normalized by a robust calibration scale so neither dominates.

  Localization:
    tau_star(x_star) = weighted_quantile(cal_scores, w(x_star), q=1-alpha)
  where weights w(x_star) come from an RBF on input-field features phi(x).
  """

  def __init__(
    self,
    *,
    w_l2: float = 1.0,
    w_logspec: float = 1.0,
    eps: float = 1e-8,
    eps_spec: float = 1e-12,
    # localization feature function
    phi_x_fn=default_x_features,
    phi_out_hw=(8, 8),
    phi_eps: float = 1e-6,
  ):
    self.w_l2 = float(w_l2)
    self.w_logspec = float(w_logspec)
    self.eps = float(eps)
    self.eps_spec = float(eps_spec)
    self.phi_x_fn = phi_x_fn
    self.phi_out_hw = tuple(phi_out_hw)
    self.phi_eps = float(phi_eps)

  # ---- base terms
  def _term_l2(self, y: jnp.ndarray, yhat: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.mean((y - yhat) ** 2) + self.eps)

  def _term_logspec(self, y: jnp.ndarray, yhat: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(_logspec_dist(y, yhat, eps=self.eps_spec) + self.eps)

  def _terms(self, y: jnp.ndarray, yhat: jnp.ndarray):
    return {
      "l2": self._term_l2(y, yhat),
      "logspec": self._term_logspec(y, yhat),
    }

  # ---- fit global scales + (global) tau
  def fit(self, y: jnp.ndarray, yhat: jnp.ndarray, alpha: jnp.ndarray):
    """
    y,yhat: (N,H,W,C)
    alpha: scalar or (A,) array in [0,1]
    """
    self.alpha = alpha
    self.n = y.shape[0]
    self.k_alpha = jnp.ceil((1.0 - self.alpha) * (self.n + 1)).astype(int)

    # term scales (robust) from calibration pairs
    terms = jax.vmap(self._terms, in_axes=(0, 0))(y, yhat)
    self.scales = {
      "l2": _robust_scale(terms["l2"]),
      "logspec": _robust_scale(terms["logspec"]),
    }

    self.cal_scores = self.batch_score(y, yhat)          # (N,)
    self.tau = jnp.sort(self.cal_scores)[self.k_alpha]   # scalar or (A,)

  # ---- scoring
  def _score(self, y: jnp.ndarray, yhat: jnp.ndarray) -> jnp.ndarray:
    """
    y,yhat: (H,W,C) -> scalar
    """
    t = self._terms(y, yhat)

    l2_n = t["l2"] / self.scales["l2"]
    sp_n = t["logspec"] / self.scales["logspec"]

    num = self.w_l2 * (l2_n ** 2) + self.w_logspec * (sp_n ** 2)
    den = (self.w_l2 + self.w_logspec + self.eps)
    return jnp.sqrt(num / den + self.eps)

  def batch_score(self, y: jnp.ndarray, yhat: jnp.ndarray) -> jnp.ndarray:
    # paired calibration: (N,H,W,C)->(N,)
    return jax.vmap(self._score, in_axes=(0, 0))(y, yhat)

  # Important: y is mapped, yhat is treated as "static" (shared) for repeated sampling.
  score      = jax.vmap(_score, (None, 0, None))
  score_grad = jax.vmap(jax.value_and_grad(_score, argnums=1), (None, 0, None))

  # ---- localization around x only
  def fit_localizer(self, x_cal: jnp.ndarray):
    """
    Precompute Phi(x) for calibration inputs and standardize it.
    x_cal: (N,H,W,Cx)
    """
    Phi = jax.vmap(lambda x: self.phi_x_fn(x, out_hw=self.phi_out_hw, eps=self.phi_eps))(x_cal)
    self.Phi_cal = Phi
    self.phi_mu, self.phi_sd = fit_phi_standardizer(Phi)

  def tau_weighted(self, x_star: jnp.ndarray, *, alpha: float, h: float = 1.0) -> jnp.ndarray:
    """
    Return localized threshold tau(x_star) using weighted quantiles of *calibration scores*.

    Requires:
      - fit(...) already called (to set cal_scores)
      - fit_localizer(x_cal) already called (to set Phi_cal, mu, sd)

    x_star: (H,W,Cx)
    alpha: scalar in [0,1]
    """
    phi_star = self.phi_x_fn(x_star, out_hw=self.phi_out_hw, eps=self.phi_eps)
    w = rbf_weights(self.Phi_cal, phi_star, self.phi_mu, self.phi_sd, h=h)
    return weighted_quantile(self.cal_scores, w, q=1.0 - alpha)
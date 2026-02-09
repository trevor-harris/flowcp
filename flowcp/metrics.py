import jax
import jax.numpy as jnp

def extract_patches_grid(y: jnp.ndarray, p: int, stride: int) -> jnp.ndarray:
  """
  Extract (overlapping) p×p patches on a regular grid.

  Args:
    y: (H, W, C)
    p: patch size
    stride: step between patch top-left corners

  Returns:
    P: (N, D) where D = p*p*C and N ≈ ((H-p)/stride+1)*((W-p)/stride+1)
  """
  H, W, C = y.shape
  ys = jnp.arange(0, H - p + 1, stride)
  xs = jnp.arange(0, W - p + 1, stride)

  def row(i):
    def col(j):
      patch = jax.lax.dynamic_slice(y, (i, j, 0), (p, p, C))
      return patch.reshape(-1)
    return jax.vmap(col)(xs)

  P = jax.vmap(row)(ys)  # (Ny, Nx, D)
  return P.reshape(-1, p * p * C)


def _normalize_patches(P: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
  """
  Per-patch standardization: (patch - mean) / std.
  This focuses the comparison on texture/structure rather than absolute intensity.
  """
  mu = jnp.mean(P, axis=1, keepdims=True)
  sd = jnp.std(P, axis=1, keepdims=True)
  return (P - mu) / (sd + eps)


def _sq_dists(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
  """
  Pairwise squared Euclidean distances.
  A: (N,D), B: (M,D) -> (N,M)
  """
  a2 = jnp.sum(A * A, axis=1, keepdims=True)  # (N,1)
  b2 = jnp.sum(B * B, axis=1, keepdims=True)  # (M,1)
  return jnp.maximum(a2 + b2.T - 2.0 * (A @ B.T), 0.0)


def mmd2_rbf(A: jnp.ndarray, B: jnp.ndarray, sigma: float, unbiased: bool = True) -> jnp.ndarray:
  """
  Squared MMD with an RBF kernel k(a,b)=exp(-||a-b||^2 / sigma^2).

  unbiased=True removes diagonal terms in Kxx, Kyy, which reduces small-sample bias.
  """
  N = A.shape[0]
  M = B.shape[0]
  inv = 1.0 / (sigma * sigma)

  Dxx = _sq_dists(A, A)
  Dyy = _sq_dists(B, B)
  Dxy = _sq_dists(A, B)

  Kxx = jnp.exp(-inv * Dxx)
  Kyy = jnp.exp(-inv * Dyy)
  Kxy = jnp.exp(-inv * Dxy)

  if unbiased:
    Kxx = Kxx - jnp.eye(N, dtype=Kxx.dtype)
    Kyy = Kyy - jnp.eye(M, dtype=Kyy.dtype)
    mmd_xx = jnp.sum(Kxx) / (N * (N - 1) + 1e-16)
    mmd_yy = jnp.sum(Kyy) / (M * (M - 1) + 1e-16)
  else:
    mmd_xx = jnp.mean(Kxx)
    mmd_yy = jnp.mean(Kyy)

  mmd_xy = jnp.mean(Kxy)
  return mmd_xx + mmd_yy - 2.0 * mmd_xy

def mmd_score(
  y_samps: jnp.ndarray,            # (K, H, W, C)
  y_true: jnp.ndarray,             # (H, W, C)
  *,
  p: int = 7,
  stride: int = 4,
  sigmas: jnp.ndarray = jnp.array([0.5, 1.0, 2.0], dtype=jnp.float32),
  normalize: bool = True,
  unbiased: bool = True,
  max_patches_true: int = 2048,
  max_patches_samps: int = 4096,
  eps_norm: float = 1e-6,
) -> jnp.ndarray:
  """
  Patch-MMD^2 using grid patches.

  We compare:
    A = patches(y_true)
    B = pooled patches over all K samples

  Args:
    p, stride: patch geometry
    sigmas: RBF bandwidth(s); we average MMD^2 across these sigmas for stability
    normalize: per-patch standardization (recommended for “texture” comparison)
    unbiased: unbiased MMD estimator
    max_patches_*: truncation to bound O(N^2) kernel cost

  Returns:
    scalar MMD^2 (lower is better; 0 means identical patch distributions)
  """
  # Truth patches
  A = extract_patches_grid(y_true, p, stride)
  A = A[: jnp.minimum(A.shape[0], max_patches_true)]

  # Sample patches, pooled over K
  def one(ys):
    return extract_patches_grid(ys, p, stride)

  B = jax.vmap(one)(y_samps)                # (K, Ns, D)
  B = B.reshape(-1, B.shape[-1])            # (K*Ns, D)
  B = B[: jnp.minimum(B.shape[0], max_patches_samps)]

  if normalize:
    A = _normalize_patches(A, eps=eps_norm)
    B = _normalize_patches(B, eps=eps_norm)

  # Multi-bandwidth MMD (more robust than picking one sigma)
  def one_sigma(s):
    return jnp.clip(mmd2_rbf(A, B, sigma=s, unbiased=unbiased), 0.0)

  return jnp.sqrt(jnp.mean(jax.vmap(one_sigma)(sigmas)))

@jax.jit
def lsd_score(y_samps, y_true, eps=1e-12):
  y_samps = y_samps.squeeze()   # (K,H,W)
  y_true  = y_true.squeeze()    # (H,W)

  def compute_beta(u):
    F = jnp.fft.rfft2(u - jnp.mean(u))
    S = jnp.abs(F)**2                          # power
    return jnp.mean(S)                  # log spectrum

  def amp_spec(u):
    F = jnp.fft.rfft2(u - jnp.mean(u))
    S = jnp.abs(F)**2                          # power
    return jnp.log1p(S/beta)                   # log spectrum

  beta = compute_beta(y_true)
  S_true  = amp_spec(y_true)[None, ...]
  S_samps = jax.vmap(amp_spec)(y_samps)

  return jnp.mean((S_true - S_samps)**2)

@jax.jit
def energy_score(y_samps, y_true, eps=1e-12):
  K = y_samps.shape[0]
  X = y_samps.reshape(K, -1)
  y = y_true.reshape(-1)
  D = X.shape[1]

  d1 = jnp.sqrt(jnp.mean((X - y[None, :])**2, axis=1) + eps)   # RMS
  term1 = jnp.mean(d1)

  s = jnp.mean(X * X, axis=1, keepdims=True)                  # mean square, not sum
  dist2 = jnp.maximum(s + s.T - 2.0 * (X @ X.T) / D, 0.0)     # normalize dot by D
  d2 = jnp.sqrt(dist2 + eps)
  term2 = 0.5 * jnp.mean(d2)

  return term1 - term2
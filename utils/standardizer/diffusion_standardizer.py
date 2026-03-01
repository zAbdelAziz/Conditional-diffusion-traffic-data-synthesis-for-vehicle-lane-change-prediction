from numpy import ndarray, array, arange, sqrt, where, zeros, ones, float32


class DiffusionStandardizer:
	def __init__(self, vision_R: float, tol: float = 1e-3):
		"""
		- Fits ego stats on all ego_xy
		- Fits neighbor dx/dy stats ONLY on PRESENT neighbors (r < R - tol)
		- After transform / inverse_transform, re-imposes missing sentinel exactly.
		"""
		self.vision_R = float(vision_R)
		self.tol = float(tol)

	@staticmethod
	def _slot_signs() -> ndarray:
		return array([+1, -1, +1, -1, +1, -1], dtype=float32)

	def _absent_mask(self, X14: ndarray) -> ndarray:
		# X14: [N,T,14]
		nbr = X14[:, :, 2:14].reshape(X14.shape[0], X14.shape[1], 6, 2).astype(float32)
		r = sqrt(nbr[..., 0] ** 2 + nbr[..., 1] ** 2).astype(float32)
		return r >= (self.vision_R - self.tol)  # [N,T,6] bool

	def fit(self, X: ndarray, idx: ndarray):
		assert X.ndim == 3 and X.shape[-1] == 14, f"expected [N,T,14], got {X.shape}"
		Xtr = X[idx].astype(float32, copy=False)

		mu = zeros((14,), dtype=float32)
		sigma = ones((14,), dtype=float32)

		# ego dims
		ego = Xtr[:, :, 0:2].reshape(-1, 2)
		mu_ego = ego.mean(axis=0).astype(float32)
		sd_ego = ego.std(axis=0).astype(float32)
		sd_ego[sd_ego < 1e-6] = 1.0
		mu[0:2] = mu_ego
		sigma[0:2] = sd_ego

		# neighbor dims: present-only
		nbr = Xtr[:, :, 2:14].reshape(-1, 6, 2)  # [Ntr*T,6,2]
		r = sqrt(nbr[..., 0] ** 2 + nbr[..., 1] ** 2)
		present = r < (self.vision_R - self.tol)

		if present.any():
			nx = nbr[..., 0][present]
			ny = nbr[..., 1][present]
			mu_nx = float(nx.mean()) if nx.size else 0.0
			mu_ny = float(ny.mean()) if ny.size else 0.0
			sd_nx = float(nx.std()) if nx.size else 1.0
			sd_ny = float(ny.std()) if ny.size else 1.0
			if sd_nx < 1e-6: sd_nx = 1.0
			if sd_ny < 1e-6: sd_ny = 1.0
		else:
			mu_nx = mu_ny = 0.0
			sd_nx = sd_ny = 1.0

		# apply same (mu,sigma) to all neighbor x dims and all neighbor y dims
		nbr_dims = arange(2, 14)
		x_dims = nbr_dims[0::2]
		y_dims = nbr_dims[1::2]
		mu[x_dims] = float32(mu_nx)
		mu[y_dims] = float32(mu_ny)
		sigma[x_dims] = float32(sd_nx)
		sigma[y_dims] = float32(sd_ny)

		return mu, sigma

	def transform(self, X: ndarray, mu: ndarray, sigma: ndarray):
		"""
		X: [N,T,14] raw diffusion representation (missing neighbors encoded as (0, ±R))
		Returns standardized Y with missing sentinel re-imposed in standardized units.
		"""
		assert X.ndim == 3 and X.shape[-1] == 14, f"expected [N,T,14], got {X.shape}"
		X = X.astype(float32, copy=False)
		mu = mu.astype(float32, copy=False)
		sigma = sigma.astype(float32, copy=False)

		# detect absent on RAW X (stable)
		absent = self._absent_mask(X)  # [N,T,6] bool

		# standardize full tensor
		Y = (X - mu[None, None, :]) / sigma[None, None, :]

		# neighbor view
		Ynbr = Y[:, :, 2:14].reshape(X.shape[0], X.shape[1], 6, 2)  # [N,T,6,2]
		dx = Ynbr[..., 0]  # [N,T,6]
		dy = Ynbr[..., 1]  # [N,T,6]

		signs = self._slot_signs().astype(float32)  # [6]

		# Your fit() ties all neighbor-x dims together and all neighbor-y dims together,
		# so taking mu[2],sigma[2] and mu[3],sigma[3] is consistent.
		mu_nx, sig_nx = float(mu[2]), float(sigma[2])
		mu_ny, sig_ny = float(mu[3]), float(sigma[3])

		# standardized sentinel values (broadcastable to [N,T,6])
		dx_sentinel = float32((0.0 - mu_nx) / sig_nx)  # scalar
		dy_sentinel = ((signs * self.vision_R) - mu_ny) / sig_ny  # [6]

		# impose sentinel using where (broadcast-safe)
		dx[:] = where(absent, dx_sentinel, dx)
		dy[:] = where(absent, dy_sentinel[None, None, :], dy)

		# write back
		Y[:, :, 2:14] = Ynbr.reshape(X.shape[0], X.shape[1], 12)
		return Y.astype(float32, copy=False)

	def inverse_transform(self, X: ndarray, mu: ndarray, sigma: ndarray):
		"""
		X: [N,T,14] standardized diffusion representation
		Returns raw-space Y with missing sentinel re-imposed exactly as (0, ±R).
		"""
		assert X.ndim == 3 and X.shape[-1] == 14, f"expected [N,T,14], got {X.shape}"
		X = X.astype(float32, copy=False)
		mu = mu.astype(float32, copy=False)
		sigma = sigma.astype(float32, copy=False)

		# de-standardize
		Y = X * sigma[None, None, :] + mu[None, None, :]

		# detect absent on RAW-space Y
		absent = self._absent_mask(Y)  # [N,T,6] bool

		Ynbr = Y[:, :, 2:14].reshape(Y.shape[0], Y.shape[1], 6, 2)
		dx = Ynbr[..., 0]  # [N,T,6]
		dy = Ynbr[..., 1]  # [N,T,6]

		signs = self._slot_signs().astype(float32)  # [6]
		dy_sentinel_raw = (signs * self.vision_R).astype(float32)  # [6]

		dx[:] = where(absent, 0.0, dx)
		dy[:] = where(absent, dy_sentinel_raw[None, None, :], dy)

		Y[:, :, 2:14] = Ynbr.reshape(Y.shape[0], Y.shape[1], 12)
		return Y.astype(float32, copy=False)
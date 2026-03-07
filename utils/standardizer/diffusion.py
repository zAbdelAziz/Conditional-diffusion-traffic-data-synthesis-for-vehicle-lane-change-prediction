from numpy import ndarray, zeros, ones, float32


class DiffusionStandardizer:
	def __init__(self, p_present_threshold: float = 0.5):
		self.p_present_threshold = float(p_present_threshold)

	@staticmethod
	def _slot_triplet_dims(slot: int):
		base = 2 + 3 * slot
		return base, base + 1, base + 2  # dx, dy, p

	def _split_neighbor_block(self, X20: ndarray):
		nbr = X20[:, :, 2:20].reshape(X20.shape[0], X20.shape[1], 6, 3).astype(float32)
		dx = nbr[..., 0]
		dy = nbr[..., 1]
		p = nbr[..., 2]
		return nbr, dx, dy, p

	def _present_mask(self, X20: ndarray) -> ndarray:
		_, _, _, p = self._split_neighbor_block(X20)
		return p > self.p_present_threshold

	def fit(self, X: ndarray, idx: ndarray):
		assert X.ndim == 3 and X.shape[-1] == 20, f"expected [N,T,20], got {X.shape}"
		Xtr = X[idx].astype(float32, copy=False)

		mu = zeros((20,), dtype=float32)
		sigma = ones((20,), dtype=float32)

		# ego dims [vx, vy]
		ego = Xtr[:, :, 0:2].reshape(-1, 2)
		mu_ego = ego.mean(axis=0).astype(float32)
		sd_ego = ego.std(axis=0).astype(float32)
		sd_ego[sd_ego < 1e-6] = 1.0
		mu[0:2] = mu_ego
		sigma[0:2] = sd_ego

		# slot-wise neighbor stats
		nbr, dx, dy, p = self._split_neighbor_block(Xtr)

		for s in range(6):
			dx_dim, dy_dim, p_dim = self._slot_triplet_dims(s)

			present_s = p[:, :, s] > self.p_present_threshold

			if present_s.any():
				dx_s = dx[:, :, s][present_s]
				dy_s = dy[:, :, s][present_s]

				mu_dx = float(dx_s.mean()) if dx_s.size else 0.0
				mu_dy = float(dy_s.mean()) if dy_s.size else 0.0
				sd_dx = float(dx_s.std()) if dx_s.size else 1.0
				sd_dy = float(dy_s.std()) if dy_s.size else 1.0

				if sd_dx < 1e-6:
					sd_dx = 1.0
				if sd_dy < 1e-6:
					sd_dy = 1.0
			else:
				mu_dx, mu_dy = 0.0, 0.0
				sd_dx, sd_dy = 1.0, 1.0

			# apply same (mu,sigma) to all neighbor x dims and all neighbor y dims
			mu[dx_dim] = float32(mu_dx)
			mu[dy_dim] = float32(mu_dy)
			sigma[dx_dim] = float32(sd_dx)
			sigma[dy_dim] = float32(sd_dy)
			mu[p_dim] = 0.0
			sigma[p_dim] = 1.0

		return mu, sigma

	def transform(self, X: ndarray, mu: ndarray, sigma: ndarray):
		assert X.ndim == 3 and X.shape[-1] == 20, f"expected [N,T,20], got {X.shape}"
		X = X.astype(float32, copy=False)
		mu = mu.astype(float32, copy=False)
		sigma = sigma.astype(float32, copy=False)

		Y = (X - mu[None, None, :]) / sigma[None, None, :]

		# detect absent on RAW X (stable)
		absent = ~self._present_mask(X)

		# neighbor view
		Ynbr, dx, dy, p = self._split_neighbor_block(Y)
		dx[absent] = 0.0
		dy[absent] = 0.0
		p[:] = p.clip(0.0, 1.0)

		Y[:, :, 2:20] = Ynbr.reshape(X.shape[0], X.shape[1], 18)
		return Y.astype(float32, copy=False)

	def inverse_transform(self, X: ndarray, mu: ndarray, sigma: ndarray):
		assert X.ndim == 3 and X.shape[-1] == 20, f"expected [N,T,20], got {X.shape}"
		X = X.astype(float32, copy=False)
		mu = mu.astype(float32, copy=False)
		sigma = sigma.astype(float32, copy=False)

		# de-standardize
		Y = X * sigma[None, None, :] + mu[None, None, :]

		Ynbr, dx, dy, p = self._split_neighbor_block(Y)

		absent = p <= self.p_present_threshold
		dx[absent] = 0.0
		dy[absent] = 0.0
		p[:] = p.clip(0.0, 1.0)

		Y[:, :, 2:20] = Ynbr.reshape(Y.shape[0], Y.shape[1], 18)
		return Y.astype(float32, copy=False)
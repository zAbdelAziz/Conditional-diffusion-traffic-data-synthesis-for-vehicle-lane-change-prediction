from numpy import ndarray, float32


class DownstreamStandardizer:
	def fit(self, X: ndarray, idx: ndarray):
		# [Ntr*T, D]
		Xf = X[idx].reshape(-1, X.shape[-1]).astype(float32)

		mu = Xf.mean(axis=0).astype(float32)
		sigma = Xf.std(axis=0).astype(float32)
		sigma[sigma < 1e-6] = 1.0

		return mu, sigma

	def transform(self, X: ndarray, mu: ndarray, sigma: ndarray):
		Y = X.astype(float32, copy=True)
		Y = (Y - mu[None, None, :]) / sigma[None, None, :]
		return Y

	def inverse_transform(self, X: ndarray, mu: ndarray, sigma: ndarray):
		Y = X.astype(float32, copy=True)
		Y = Y * sigma[None, None, :] + mu[None, None, :]
		return Y

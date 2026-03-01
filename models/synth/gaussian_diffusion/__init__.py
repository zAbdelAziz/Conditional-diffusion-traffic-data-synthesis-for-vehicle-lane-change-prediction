from torch.nn import Module

from torch import Tensor, tensor, zeros, full, arange, linspace, randn, randn_like, unique_consecutive, no_grad, cat, flip, clamp
from torch import sqrt, cos, pi, cumprod, float32, long
from torch import round as torch_round


class GaussianDiffusionModel(Module):
	def __init__(self, T: int, beta_schedule: str = "cosine"):
		super().__init__()
		# Number of diffusion timesteps
		self.T = T

		# Build the beta schedule as a 1D tensor of length shape: [T]
		betas = self._make_betas(T, beta_schedule)
		# Convert betas to per-step alphas: alpha_t = 1 - beta_t
		alphas = 1.0 - betas
		# Compute cumulative product alpha_bar_t = PI_{i=0}^t alpha_i
		# signal power at timestep t
		alpha_bar = cumprod(alphas, dim=0)

		# Non-Trainable Buffers
		self.register_buffer("betas", betas)
		self.register_buffer("alphas", alphas)
		self.register_buffer("alpha_bar", alpha_bar)

	@staticmethod
	def _make_betas(T: int, schedule: str):
		if schedule == "linear":
			# create T linearly spaced betas from 1e-4 to 0.02 (common DDPM baseline)
			return linspace(1e-4, 0.02, T)
		if schedule == "cosine":
			# Small offset (from cosine schedule literature)
			# to avoid singular behavior near t=0 and to shape the curve slightly
			s = 0.008
			# Create timestep indices [0, 1, ..., T] (T+1 points) as float32
			steps = arange(T + 1, dtype=float32)
			# Compute squared cosine curve over normalized time in [0,1] and shifted by s
			# produces alpha_bar-like unnormalized values
			# f = cos[(pi/2) * ((s + steps/t) / (1+s))]^2
			f = cos(((steps / T) + s) / (1 + s) * pi / 2) ** 2
			# Normalize so alpha_bar[0] == 1 (cumulative product of alphas starts at 1)
			alpha_bar = f / f[0]
			# Convert cumulative alpha_bar into per-step beta
			# beta_t = 1 - (alpha_bar_t / alpha_bar_{t-1})
			betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
			# Clamp for numerical stability
			# avoid exactly 0 (no noise) and too close to 1 (blows up variance)
			return betas.clamp(1e-6, 0.999)
		raise ValueError(f"Unknown beta schedule: {schedule}")

	def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor):
		"""Sample from the forward diffusion q(x_t | x_0) given x0, timestep indices t, and provided noise"""
		# Gather alpha_bar for each batch element at its timestep and reshape for broadcasting over (C, L)/(C, H, W)/etc
		a_bar = self.alpha_bar[t].view(-1, 1, 1)
		# Apply the closed-form forward diffusion: x_t = sqrt(a_bar)*x0 + sqrt(1-a_bar)*noise
		return sqrt(a_bar) * x0 + sqrt(1.0 - a_bar) * noise

	@no_grad()
	def p_sample_loop(self, denoiser, shape, y=None, device=None, *, steps: int = None, method: str = "ddim", eta: float = 0.0):
		device = device or self.betas.device

		# Initialize x at time T with standard Gaussian noise: x_T ~ N(0, I)
		x = randn(shape, device=device)

		method = str(method).lower()
		# Build the reverse-time indices: either full schedule or a respaced schedule
		if steps is None:
			ts = arange(self.T - 1, -1, -1, device=device, dtype=long)
		else:
			ts = self._make_respaced_schedule(int(steps), device=device)

		# Loop over reversed timesteps
		for i in range(ts.numel()):
			# Current timestep
			ti = int(ts[i].item())
			# batch-shaped timestep tensor for the denoiser, all elements equal to ti
			t = full((shape[0],), ti, device=device, dtype=long)

			# Predict noise eps_hat = eps_theta(x_t, t, y) using the provided denoiser
			eps_hat = denoiser(x, t, y)

			# Fetch alpha_bar(t) as a scalar (single timestep)
			a_bar_t = self.alpha_bar[ti]

			# Precompute sqrt(alpha_bar_t) for reuse
			sqrt_a_bar_t = sqrt(a_bar_t)
			# Precompute sqrt(1 - alpha_bar_t) for reuse
			sqrt_one_minus_a_bar_t = sqrt(1.0 - a_bar_t)

			# Estimate x0 using the standard DDPM inversion
			# x0_hat = (x_t - sqrt(1-a_bar_t)*eps_hat) / sqrt(a_bar_t)
			x0_hat = (x - sqrt_one_minus_a_bar_t * eps_hat) / (sqrt_a_bar_t + 1e-12)

			# Determine next timestep index in the (possibly respaced) schedule
			# last iteration maps to final output
			if i == ts.numel() - 1:
				t_next = -1
			else:
				t_next = int(ts[i + 1].item())

			# Exit Loop If there is no next timestep
			if t_next < 0:
				x = x0_hat
				continue

			# Fetch alpha_bar at the next timestep
			a_bar_next = self.alpha_bar[t_next]
			# Precompute sqrt(alpha_bar_next)
			sqrt_a_bar_next = sqrt(a_bar_next)
			# Precompute sqrt(1 - alpha_bar_next)
			sqrt_one_minus_a_bar_next = sqrt(1.0 - a_bar_next)

			if method == "ddim":
				# DDIM update:
				# x_{t_next} = sqrt(a_bar_next)*x0_hat + sqrt(1-a_bar_next - sigma^2)*eps_hat + sigma*z
				# sigma = eta * sqrt((1-a_bar_next)/(1-a_bar_t)) * sqrt(1 - a_bar_t/a_bar_next)
				# sigma controls injected noise (eta=0 => deterministic)
				if eta > 0.0:
					# Compute ratio term (1-a_bar_next)/(1-a_bar_t)
					frac = (1.0 - a_bar_next) / (1.0 - a_bar_t + 1e-12)
					# Compute 1 - a_bar_t/a_bar_next
					term = 1.0 - (a_bar_t / (a_bar_next + 1e-12))
					# sigma = eta * sqrt(frac * term)
					# clamped to avoid negative under sqrt due to floating error
					sigma = eta * sqrt(clamp(frac * term, min=0.0))
				else:
					# If eta==0 then sigma is exactly zero
					sigma = tensor(0.0, device=device)

				# Compute coefficient for eps_hat: c = sqrt(1 - a_bar_next - sigma^2)
				# Sqrt should be >= 0 to avoid imaginary number bug
				c = sqrt(clamp(1.0 - a_bar_next - sigma**2, min=0.0))

				# stochastic DDIM should sample fresh noise z and include sigma*z term
				if eta > 0.0:
					z = randn_like(x)
					x = sqrt_a_bar_next * x0_hat + c * eps_hat + sigma * z
				# If deterministic DDIM the noise term is omitted
				else:
					x = sqrt_a_bar_next * x0_hat + c * eps_hat

			elif method == "ddpm":
				# DDPM-like step but respaced:
				# use beta_t from original index ti
				# approx when skipping steps
				beta_t = self.betas[ti]
				alpha_t = self.alphas[ti]

				# Compute mean of p(x_{t-1}|x_t) under the eps-parameterization
				# mu = 1/sqrt(alpha_t) * (x_t - (beta_t / sqrt(1-a_bar_t)) * eps_hat)
				mu = (1.0 / sqrt(alpha_t)) * (x - (beta_t / (sqrt_one_minus_a_bar_t + 1e-12)) * eps_hat)

				# Sample noise for the stochastic reverse step
				z = randn_like(x)
				# Use sigma = sqrt(beta_t) as a simple variance choice for the noise term
				sigma = sqrt(beta_t)
				# Draw x_{t-1} = mu + sigma*z
				x = mu + sigma * z
			else:
				raise ValueError("method must be 'ddim' or 'ddpm'")

		return x

	def _make_respaced_schedule(self, steps: int, device):
		steps = int(steps)
		if steps <= 0:
			raise ValueError("steps must be > 0")

		# If requested steps are >= original T I just use the full schedule: T-1, ..., 0
		if steps >= self.T:
			return arange(self.T - 1, -1, -1, device=device, dtype=long)

		# evenly spaced indices in [0, T-1]
		t = linspace(0, self.T - 1, steps, device=device)
		# Round to nearest integer indices and cast to int64
		t = torch_round(t).to(long)
		# Remove duplicates created by rounding (keeps first occurrence of consecutive duplicates)
		t = unique_consecutive(t)
		# If rounding collapsed everything fall back to a minimal valid schedule
		if t.numel() < 2:
			t = tensor([0, self.T - 1], device=device, dtype=long)

		# Ensure the first timestep index is 0
		if t[0].item() != 0:
			t = cat([zeros(1, device=device, dtype=long), t], dim=0)
		# Ensure the last timestep index is T-1
		if t[-1].item() != self.T - 1:
			t = cat([t, tensor([self.T - 1], device=device, dtype=long)], dim=0)

		# Return schedule in descending order for the reverse process
		# (start at high noise, go to low noise)
		return flip(t, dims=(0,))


from torch import Tensor, tensor, zeros_like, randn, randn_like, arange, empty, long, linspace, clamp, round as torch_round, unique_consecutive, zeros, flip, cat
from torch import sqrt, cos, pi, cumprod, float32
from torch.nn import Module
from torch import no_grad


class MaskedGaussianDiffusionModel(Module):
	def __init__(self, T: int, beta_schedule: str = "cosine", cfg_scale: float = 0.0):
		super().__init__()
		self.T = T
		self.cfg_scale = cfg_scale

		betas = self._make_betas(T, beta_schedule)
		alphas = 1.0 - betas
		alpha_bar = cumprod(alphas, dim=0)

		self.register_buffer("betas", betas)
		self.register_buffer("alphas", alphas)
		self.register_buffer("alpha_bar", alpha_bar)

		self.register_buffer("sqrt_alpha_bar", sqrt(alpha_bar))
		self.register_buffer("sqrt_one_minus_alpha_bar", sqrt(1.0 - alpha_bar))
		self.register_buffer("sqrt_alphas", sqrt(alphas))
		self.register_buffer("sqrt_betas", sqrt(betas))
		self.register_buffer("sqrt_recip_alphas", 1.0 / (self.sqrt_alphas + 1e-12))
		self.register_buffer("beta_over_sqrt_one_minus_alpha_bar", betas / (self.sqrt_one_minus_alpha_bar + 1e-12))

	@staticmethod
	def _make_betas(T: int, schedule: str):
		if schedule == "linear":
			return linspace(1e-4, 0.02, T)
		if schedule == "cosine":
			s = 0.008
			steps = arange(T + 1, dtype=float32)
			f = cos(((steps / T) + s) / (1 + s) * pi / 2) ** 2
			alpha_bar = f / f[0]
			betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
			return betas.clamp(1e-6, 0.999)
		raise ValueError(f"Unknown beta schedule: {schedule}")

	def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor):
		s1 = self.sqrt_alpha_bar[t].view(-1, 1, 1)
		s2 = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
		return s1 * x0 + s2 * noise

	def _denoiser_out_to_eps(self, out, x_t: Tensor, t: Tensor, gate_continuous_with_mask: bool = True):
		# Backward compatible path
		if isinstance(out, Tensor):
			return out

		if not isinstance(out, dict):
			raise TypeError(f"Unsupported denoiser output type: {type(out)}")

		B, T, D = x_t.shape
		assert D == 20, f"expected D=20, got {D}"

		eps_full = zeros_like(x_t)
		eps_full[:, :, 0:2] = out["eps_ego"]

		# [B,T,6,3]
		xt_nbr = x_t[:, :, 2:20].reshape(B, T, 6, 3)
		xt_dxdy = xt_nbr[..., 0:2]
		# [B,T,6,2]
		eps_slots = out["eps_slots"]
		# [B,T,6]
		p_hat = out["p_logits"].sigmoid().clamp(1e-4, 1.0 - 1e-4)

		# [B,1,1]
		s1 = self.sqrt_alpha_bar[t].view(-1, 1, 1)
		# [B,1,1]
		s2 = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)

		# if the mask says "absent", bias dx/dy toward clean zero
		if gate_continuous_with_mask:
			# x0 = 0 => eps = xt / sqrt(1-a_bar)
			eps_zero = xt_dxdy / (s2.unsqueeze(-1) + 1e-12)
			eps_slots = p_hat.unsqueeze(-1) * eps_slots + (1.0 - p_hat.unsqueeze(-1)) * eps_zero

		# Convert clean-mask prediction p_hat into epsilon for the p channels
		eps_p = (xt_nbr[..., 2] - s1 * p_hat) / (s2 + 1e-12)

		nbr_eps = eps_full[:, :, 2:20].reshape(B, T, 6, 3)
		nbr_eps[..., 0:2] = eps_slots
		nbr_eps[..., 2] = eps_p
		eps_full[:, :, 2:20] = nbr_eps.reshape(B, T, 18)

		return eps_full

	def predict_eps(self, denoiser, x_t: Tensor, t: Tensor, y: Tensor = None, *, gate_continuous_with_mask: bool = True):
		out_cond = denoiser(x_t, t, y)
		eps_cond = self._denoiser_out_to_eps(out_cond, x_t, t, gate_continuous_with_mask=gate_continuous_with_mask)

		if (self.cfg_scale <= 0.0) or (y is None):
			return eps_cond

		out_uncond = denoiser(x_t, t, None)
		eps_uncond = self._denoiser_out_to_eps(out_uncond, x_t, t, gate_continuous_with_mask=gate_continuous_with_mask)

		return eps_uncond + self.cfg_scale * (eps_cond - eps_uncond)

	@no_grad()
	def p_sample_loop(self, denoiser, shape, y=None, device=None, *, steps: int = None, method: str = "ddpm", eta: float = 0.0,
					  gate_continuous_with_mask: bool = True):
		device = device or self.betas.device
		x = randn(shape, device=device)

		method = str(method).lower()
		if steps is None:
			ts = arange(self.T - 1, -1, -1, device=device, dtype=long)
		else:
			ts = self._make_respaced_schedule(int(steps), device=device)

		t_buf = empty((shape[0],), device=device, dtype=long)

		for i in range(ts.numel()):
			ti = int(ts[i].item())
			t_buf.fill_(ti)

			eps_hat = self.predict_eps(denoiser, x, t_buf, y=y, gate_continuous_with_mask=gate_continuous_with_mask)

			a_bar_t = self.alpha_bar[ti]
			x0_hat = (x - self.sqrt_one_minus_alpha_bar[ti] * eps_hat) / (self.sqrt_alpha_bar[ti] + 1e-12)

			if i == ts.numel() - 1:
				t_next = -1
			else:
				t_next = int(ts[i + 1].item())

			if t_next < 0:
				x = x0_hat
				continue

			a_bar_next = self.alpha_bar[t_next]

			if method == "ddim":
				if eta > 0.0:
					frac = (1.0 - a_bar_next) / (1.0 - a_bar_t + 1e-12)
					term = 1.0 - (a_bar_t / (a_bar_next + 1e-12))
					sigma = eta * sqrt(clamp(frac * term, min=0.0))
				else:
					sigma = tensor(0.0, device=device)

				c = sqrt(clamp(1.0 - a_bar_next - sigma ** 2, min=0.0))
				if eta > 0.0:
					z = randn_like(x)
					x = self.sqrt_alpha_bar[t_next] * x0_hat + c * eps_hat + sigma * z
				else:
					x = self.sqrt_alpha_bar[t_next] * x0_hat + c * eps_hat

			elif method == "ddpm":
				mu = self.sqrt_recip_alphas[ti] * (x - self.beta_over_sqrt_one_minus_alpha_bar[ti] * eps_hat)
				z = randn_like(x)
				x = mu + self.sqrt_betas[ti] * z
			else:
				raise ValueError("method must be 'ddim' or 'ddpm'")

		return x

	def _make_respaced_schedule(self, steps: int, device):
		steps = int(steps)
		if steps <= 0:
			raise ValueError("steps must be > 0")
		if steps >= self.T:
			return arange(self.T - 1, -1, -1, device=device, dtype=long)

		t = linspace(0, self.T - 1, steps, device=device)
		t = torch_round(t).to(long)
		t = unique_consecutive(t)
		if t.numel() < 2:
			t = tensor([0, self.T - 1], device=device, dtype=long)

		if t[0].item() != 0:
			t = cat([zeros(1, device=device, dtype=long), t], dim=0)
		if t[-1].item() != self.T - 1:
			t = cat([t, tensor([self.T - 1], device=device, dtype=long)], dim=0)

		return flip(t, dims=(0,))
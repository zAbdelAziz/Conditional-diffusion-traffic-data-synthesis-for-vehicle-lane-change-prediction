from typing import Any, Optional, Tuple

from torch import Tensor, arange, empty, zeros, rand, randn, randn_like, full, full_like, linspace, no_grad
from torch import where, cumprod, pi, clamp, cos, sqrt, long, float32

from torch.nn import Module

from models.synth.hybrid_diffusion.utils import expand_time_indexed, empty_like_1d


class HybridDiffusionModel(Module):
	"""
	Hybrid diffusion model with:
	  - Gaussian diffusion on continuous channels
	  - Discrete absorbing-state diffusion on mask channels

	Denoiser contract:
		denoiser(x_t, t, y) -> {
			"eps_cont": Tensor [B, T_seq, 14],
			"mask_logits": Tensor [B, T_seq, 6],
		}

	Full state layout (D = 20):
		x[..., 0:2]    = ego continuous
		x[..., 2:20]   = 6 neighbor slots x 3 values
						 each slot = [dx, dy, p]

	Continuous subset:
		ego (2) + nbr dx/dy (6*2) = 14 dims

	Mask subset:
		nbr p for 6 slots = 6 dims

	Noisy mask state values:
		0 = absent
		1 = present
		2 = unknown / absorbed

	Numeric encoding inside full x_t for the mask channels:
		absent  -> 0.0
		present -> 1.0
		unknown -> 0.5
	"""

	MASK_ABSENT = 0
	MASK_PRESENT = 1
	MASK_UNKNOWN = 2

	def __init__(self, T: int, beta_schedule: str = "cosine", gamma_schedule: str = "cosine",
				 cfg_scale: float = 0.0, cfg_p_sample: bool = False, mask_unknown_value: float = 0.5):
		super().__init__()

		if T <= 0:
			raise ValueError(f"T must be > 0, got {T}")

		self.T = int(T)
		self.cfg_scale = float(cfg_scale)
		self.cfg_p_sample = bool(cfg_p_sample)
		self.mask_unknown_value = float(mask_unknown_value)

		# Continuous Gaussian schedule
		# [T]
		betas = self._make_betas(self.T, beta_schedule)
		alphas = 1.0 - betas
		alpha_bar = cumprod(alphas, dim=0)

		self.register_buffer("betas", betas)
		self.register_buffer("alphas", alphas)
		self.register_buffer("alpha_bar", alpha_bar)

		self.register_buffer("sqrt_alpha_bar", sqrt(alpha_bar))
		self.register_buffer("sqrt_one_minus_alpha_bar", sqrt(1.0 - alpha_bar))
		self.register_buffer("sqrt_alphas", sqrt(alphas))
		self.register_buffer("sqrt_betas", sqrt(betas))
		self.register_buffer("sqrt_recip_alphas", 1.0 / (sqrt(alphas) + 1e-12))
		self.register_buffer("beta_over_sqrt_one_minus_alpha_bar", betas / (sqrt(1.0 - alpha_bar) + 1e-12))

		# Discrete absorbing-state mask schedule
		# [T]
		gammas = self._make_gammas(self.T, gamma_schedule)
		keep_probs = 1.0 - gammas
		rho_bar = cumprod(keep_probs, dim=0)

		rho_bar_prev = empty_like_1d(rho_bar)
		rho_bar_prev[0] = 1.0
		if self.T > 1:
			rho_bar_prev[1:] = rho_bar[:-1]

		eta = gammas * rho_bar_prev / clamp(1.0 - rho_bar, min=1e-12)
		eta = clamp(eta, min=0.0, max=1.0)

		self.register_buffer("gammas", gammas)
		self.register_buffer("mask_keep_probs", keep_probs)
		self.register_buffer("mask_rho_bar", rho_bar)
		self.register_buffer("mask_rho_bar_prev", rho_bar_prev)
		self.register_buffer("mask_eta", eta)

	# Schedule builders
	@staticmethod
	def _make_betas(T: int, schedule: str) -> Tensor:
		schedule = str(schedule).lower()

		if schedule == "linear":
			return linspace(1e-4, 0.02, T, dtype=float32)

		if schedule == "cosine":
			s = 0.008
			steps = arange(T + 1, dtype=float32)
			f = cos(((steps / T) + s) / (1.0 + s) * pi / 2.0) ** 2
			alpha_bar = f / f[0]
			betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
			return clamp(betas, min=1e-6, max=0.999)

		raise ValueError(f"Unknown beta schedule: {schedule}")

	@staticmethod
	def _make_gammas(T: int, schedule: str) -> Tensor:
		schedule = str(schedule).lower()

		if schedule == "linear":
			return linspace(1e-4, 0.10, T, dtype=float32)

		if schedule == "cosine":
			s = 0.008
			steps = arange(T + 1, dtype=float32)
			f = cos(((steps / T) + s) / (1.0 + s) * pi / 2.0) ** 2
			rho_bar = f / f[0]
			keep_probs = rho_bar[1:] / rho_bar[:-1]
			gammas = 1.0 - keep_probs
			return clamp(gammas, min=1e-6, max=0.999)

		raise ValueError(f"Unknown gamma schedule: {schedule}")

	# Split / merge helpers for the full x tensor
	def _split_x_tensor(self, x: Tensor) -> Tuple[Tensor, Tensor]:
		"""
		Split full x into:
			x_cont: [B, T_seq, 14]
			m_bin_or_repr: [B, T_seq, 6]

		Notes:
			- x_cont contains:
				ego(2) + each slot dx/dy(12)
			- mask tensor is taken from the 6 p-channels.
			- For clean x0, those p-channels should be binary 0/1.
			- For noised xt, those p-channels are numeric encodings of
			  discrete states: 0.0 / 1.0 / unknown_value.
		"""
		B, T_seq, D = x.shape
		if D != 20:
			raise ValueError(f"Expected x.shape[-1] == 20, got {D}")

		# [B,T,2]
		ego = x[:, :, 0:2]
		# [B,T,6,3]
		nbr = x[:, :, 2:20].reshape(B, T_seq, 6, 3)

		# [B,T,6,2]
		nbr_xy = nbr[..., 0:2]
		# [B,T,6]
		nbr_p = nbr[..., 2]

		x_cont = zeros((B, T_seq, 14), device=x.device, dtype=x.dtype)
		x_cont[:, :, 0:2] = ego
		x_cont[:, :, 2:14] = nbr_xy.reshape(B, T_seq, 12)

		return x_cont, nbr_p

	def _merge_x_tensor(self, x_cont: Tensor, mask_repr: Tensor) -> Tensor:
		"""
		Merge:
			x_cont:    [B, T_seq, 14]
			mask_repr: [B, T_seq, 6]

		back into full x: [B, T_seq, 20]
		"""
		B, T_seq, Dc = x_cont.shape
		if Dc != 14:
			raise ValueError(f"Expected x_cont.shape[-1] == 14, got {Dc}")
		if mask_repr.shape != (B, T_seq, 6):
			raise ValueError(f"Expected mask_repr shape {(B, T_seq, 6)}, got {tuple(mask_repr.shape)}")

		x = zeros((B, T_seq, 20), device=x_cont.device, dtype=x_cont.dtype)

		# ego
		x[:, :, 0:2] = x_cont[:, :, 0:2]

		# neighbors
		nbr = x[:, :, 2:20].reshape(B, T_seq, 6, 3)
		nbr[..., 0:2] = x_cont[:, :, 2:14].reshape(B, T_seq, 6, 2)
		nbr[..., 2] = mask_repr
		x[:, :, 2:20] = nbr.reshape(B, T_seq, 18)

		return x

	# Mask state conversions
	def _clean_mask_from_x(self, x: Tensor) -> Tensor:
		"""
		Extract clean binary mask m0 from x.

		Returns:
			m0_long: [B, T_seq, 6] in {0,1}
		"""
		_, p = self._split_x_tensor(x)
		return self._to_binary_mask_long(p)

	def _encode_mask_states(self, m_t: Tensor, dtype: Optional[Any] = None) -> Tensor:
		"""
		Encode discrete mask states into numeric values inserted into full x_t.

		Mapping:
			absent  -> 0.0
			present -> 1.0
			unknown -> self.mask_unknown_value
		"""
		if dtype is None:
			dtype = float32

		out = full(m_t.shape, self.mask_unknown_value, device=m_t.device, dtype=dtype)
		out = where(m_t == self.MASK_ABSENT, full_like(out, 0.0), out)
		out = where(m_t == self.MASK_PRESENT, full_like(out, 1.0), out)
		return out

	# Forward process: continuous
	def q_sample_cont(self, x0_cont: Tensor, t: Tensor, noise: Tensor) -> Tensor:
		c1 = expand_time_indexed(self.sqrt_alpha_bar, t, x0_cont)
		c2 = expand_time_indexed(self.sqrt_one_minus_alpha_bar, t, x0_cont)
		return c1 * x0_cont + c2 * noise

	# Forward process: masks
	def q_sample_mask(self, m0: Tensor, t: Tensor) -> Tensor:
		"""
		Sample discrete noisy mask states from clean binary m0.

		Args:
			m0: [B, T_seq, 6] in {0,1}
			t:  [B]

		Returns:
			m_t: [B, T_seq, 6] in {0,1,2}
		"""
		m0_bin = self._to_binary_mask_long(m0)
		keep_prob = expand_time_indexed(self.mask_rho_bar, t, m0_bin).to(dtype=float32)

		u = rand(m0_bin.shape, device=m0_bin.device, dtype=float32)
		keep = u < keep_prob

		unknown = full_like(m0_bin, self.MASK_UNKNOWN)
		m_t = where(keep, m0_bin, unknown)
		return m_t.long()

	def q_sample(self, x0: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
		"""
		Joint forward noising for the hybrid process.

		Args:
			x0:
				Clean full input [B, T_seq, 20]
			t:
				Diffusion step indices [B]
			noise:
				Optional Gaussian noise for the continuous branch only

		Returns:
			x_t:
				Full noised tensor [B, T_seq, 20]
			eps_cont_true:
				Gaussian noise used for the continuous branch [B, T_seq, 14]
			m0_true:
				Clean binary masks [B, T_seq, 6]
			m_t_discrete:
				Noisy discrete mask states [B, T_seq, 6] in {0,1,2}

		Notes:
			- Continuous channels are Gaussian noised.
			- Mask channels are replaced by encoded discrete noisy states.
			- The full x_t is what should be passed to the denoiser.
		"""
		x0_cont, _ = self._split_x_tensor(x0)
		m0_true = self._clean_mask_from_x(x0)

		if noise is None:
			noise = randn_like(x0_cont)

		x_t_cont = self.q_sample_cont(x0_cont, t, noise)
		m_t_discrete = self.q_sample_mask(m0_true, t)
		m_t_repr = self._encode_mask_states(m_t_discrete, dtype=x0.dtype)

		x_t = self._merge_x_tensor(x_t_cont, m_t_repr)
		return x_t, noise, m0_true, m_t_discrete

	# Denoiser interface
	def _split_denoiser_output(self, out: Any, x_t: Tensor) -> Tuple[Tensor, Tensor]:
		"""
		Expected denoiser output:
			{
				"eps_cont": Tensor [B, T_seq, 14],
				"mask_logits": Tensor [B, T_seq, 6],
			}
		"""
		if not isinstance(out, dict):
			raise TypeError("Denoiser output must be a dict.")

		if "eps_cont" not in out:
			raise KeyError("Denoiser output missing key 'eps_cont'")
		if "mask_logits" not in out:
			raise KeyError("Denoiser output missing key 'mask_logits'")

		eps_cont = out["eps_cont"]
		mask_logits = out["mask_logits"]

		B, T_seq, D = x_t.shape
		if eps_cont.shape != (B, T_seq, 14):
			raise ValueError(f"'eps_cont' must have shape {(B, T_seq, 14)}, got {tuple(eps_cont.shape)}")
		if mask_logits.shape != (B, T_seq, 6):
			raise ValueError(f"'mask_logits' must have shape {(B, T_seq, 6)}, got {tuple(mask_logits.shape)}")

		return eps_cont, mask_logits

	def predict_outputs(self, denoiser: Module, x_t: Tensor, t: Tensor, y: Optional[Tensor] = None):
		"""
		Run the denoiser on full x_t.

		Supports optional classifier-free guidance at sampling time.

		Returns:
			eps_cont_hat: [B, T_seq, 14]
			mask_logits:  [B, T_seq, 6]
		"""
		out_cond = denoiser(x_t, t, y)
		eps_cond, logits_cond = self._split_denoiser_output(out_cond, x_t)

		if (self.cfg_scale <= 0.0) or (y is None) or (not self.cfg_p_sample):
			return eps_cond, logits_cond

		out_uncond = denoiser(x_t, t, None)
		eps_uncond, logits_uncond = self._split_denoiser_output(out_uncond, x_t)

		eps = eps_uncond + self.cfg_scale * (eps_cond - eps_uncond)
		logits = logits_uncond + self.cfg_scale * (logits_cond - logits_uncond)
		return eps, logits

	# Reverse process: continuous branch
	def p_mean_cont(self, x_t_cont: Tensor, eps_cont: Tensor, t: Tensor) -> Tensor:
		"""
		DDPM reverse mean for continuous subset only.
		"""
		c1 = expand_time_indexed(self.sqrt_recip_alphas, t, x_t_cont)
		c2 = expand_time_indexed(self.beta_over_sqrt_one_minus_alpha_bar, t, x_t_cont)
		return c1 * (x_t_cont - c2 * eps_cont)

	# Reverse process: mask branch
	def p_mask_probs(self, mask_logits: Tensor, m_t: Tensor, t: Tensor) -> Tensor:
		"""
		Reverse probabilities for m_{t-1} given:
			- predicted clean-mask logits
			- current discrete noisy state m_t

		Args:
			mask_logits: [B, T_seq, 6]
			m_t:         [B, T_seq, 6] in {0,1,2}
			t:           [B]

		Returns:
			probs: [B, T_seq, 6, 3]
				last dim order:
					[P(absent), P(present), P(unknown)]
		"""
		pi_theta = mask_logits.sigmoid()
		eta_t = expand_time_indexed(self.mask_eta, t, m_t).to(dtype=pi_theta.dtype)

		probs = zeros((*pi_theta.shape, 3), device=pi_theta.device, dtype=pi_theta.dtype)

		is_absent = m_t == self.MASK_ABSENT
		is_present = m_t == self.MASK_PRESENT
		is_unknown = m_t == self.MASK_UNKNOWN

		# deterministic if already known
		probs[..., 0] = where(is_absent, full_like(pi_theta, 1.0), probs[..., 0])
		probs[..., 1] = where(is_present, full_like(pi_theta, 1.0), probs[..., 1])

		# derived posterior when current state is unknown
		probs[..., 0] = where(is_unknown, eta_t * (1.0 - pi_theta), probs[..., 0])
		probs[..., 1] = where(is_unknown, eta_t * pi_theta, probs[..., 1])
		probs[..., 2] = where(is_unknown, 1.0 - eta_t, probs[..., 2])

		probs = clamp(probs, min=0.0, max=1.0)
		probs = probs / clamp(probs.sum(dim=-1, keepdim=True), min=1e-12)
		return probs

	def p_sample_mask(self, mask_logits: Tensor, m_t: Tensor, t: Tensor) -> Tensor:
		"""
		Sample m_{t-1} from the reverse discrete absorbing-state kernel.
		"""
		probs = self.p_mask_probs(mask_logits, m_t, t)  # [B,T,6,3]
		flat_probs = probs.reshape(-1, 3)
		flat_samples = flat_probs.multinomial(num_samples=1).squeeze(-1)
		return flat_samples.reshape(m_t.shape).long()

	# Full reverse sampling
	@no_grad()
	def p_sample_loop(self, denoiser: Module, shape: Tuple[int, int, int], y: Optional[Tensor] = None,
					  device: Optional[Any] = None) -> Tensor:
		"""
		Sample a full x_0 from the hybrid reverse process.

		Args:
			denoiser:
				Model with signature denoiser(x_t, t, y)
			shape:
				Full output shape, e.g. (B, T_seq, 20)
			y:
				Optional labels
			device:
				Generation device

		Returns:
			x_0_hat: [B, T_seq, 20]

		Notes:
			- Continuous part starts from N(0, I)
			- Mask part starts as fully UNKNOWN
			- Reverse updates use:
				- DDPM step for continuous branch
				- derived absorbing-state reverse kernel for masks
		"""
		device = device or self.betas.device

		B, T_seq, D = shape
		if D != 20:
			raise ValueError(f"Expected shape[-1] == 20, got {D}")

		# initialize continuous latent
		x_cont = randn((B, T_seq, 14), device=device)

		# initialize masks as fully unknown
		m_t = full((B, T_seq, 6), self.MASK_UNKNOWN, device=device, dtype=long)

		t_buf = empty((B,), device=device, dtype=long)

		for ti in range(self.T - 1, -1, -1):
			t_buf.fill_(ti)

			m_repr = self._encode_mask_states(m_t, dtype=x_cont.dtype)
			x_t = self._merge_x_tensor(x_cont, m_repr)

			eps_cont_hat, mask_logits = self.predict_outputs(denoiser, x_t, t_buf, y=y)

			# continuous reverse update
			if ti == 0:
				x_cont = self.p_mean_cont(x_cont, eps_cont_hat, t_buf)
			else:
				mu = self.p_mean_cont(x_cont, eps_cont_hat, t_buf)
				sigma = expand_time_indexed(self.sqrt_betas, t_buf, x_cont)
				x_cont = mu + sigma * randn_like(x_cont)

			# mask reverse update
			if ti == 0:
				m_t = (mask_logits.sigmoid() >= 0.5).long()
			else:
				m_t = self.p_sample_mask(mask_logits, m_t, t_buf)

		m_repr = self._encode_mask_states(m_t, dtype=x_cont.dtype)
		x0_hat = self._merge_x_tensor(x_cont, m_repr)
		return x0_hat

	# Utilities
	@staticmethod
	def _to_binary_mask_long(m: Tensor) -> Tensor:
		if m.dtype == long:
			if ((m < 0) | (m > 1)).any():
				raise ValueError("Expected clean mask values in {0,1}.")
			return m.long()
		return (m > 0.5).long()

	@staticmethod
	def decode_clean_mask_from_logits(mask_logits: Tensor, threshold: float = 0.5) -> Tensor:
		return (mask_logits.sigmoid() >= float(threshold)).long()



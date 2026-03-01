from math import log

from torch import Tensor, zeros_like, cat, arange, cos, sin, exp
from torch.nn import Module


class SinusoidalTimeEmbedding(Module):
	def __init__(self, dim: int):
		super().__init__()
		self.dim = dim

	def forward(self, t: Tensor):
		device = t.device

		# Use half dim for sin and half for cos
		half = self.dim // 2
		# Create frequencies: [half], exponentially spaced
		# e^((-log(10000) * [0, dim/2]) / (0.5 * dim  - 1))
		freqs = exp(-log(10000) * arange(0, half, device=device).float() / (half - 1))

		# t [B] to [B,1] * freqs [half] to [B,half]
		args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
		# Concatenate sin and cos: [B,half] + [B,half] = [B,2*half]
		emb = cat([sin(args), cos(args)], dim=1)
		# If dim is odd pad one zero column: [B,2*half] -> [B,E]
		if self.dim % 2 == 1:
			emb = cat([emb, zeros_like(emb[:, :1])], dim=1)
		return emb
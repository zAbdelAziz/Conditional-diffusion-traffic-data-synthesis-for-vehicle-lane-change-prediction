from math import log
from torch import Tensor, zeros, arange, exp, sin, cos, einsum, stack, cat
from torch.nn import Module


class PositionalEncoding(Module):
	def __init__(self, d_model: int, max_len: int = 2048):
		super().__init__()
		pe = zeros(max_len, d_model)
		pos = arange(0, max_len).float().unsqueeze(1)
		div = exp(-log(10000) * arange(0, d_model, 2).float() / d_model)
		pe[:, 0::2] = sin(pos * div)
		pe[:, 1::2] = cos(pos * div)

		# [1, max_len, d_model]
		self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

	def forward(self, x: Tensor):
		# x: [B,T,D]
		T = x.size(1)
		return x + self.pe[:, :T, :]


class RoPE(Module):
	def __init__(self, head_dim: int, base: float = 10000.0):
		super().__init__()
		# Head Dimension should be EVEN
		self.head_dim = head_dim
		self.base = base

		# rotate an even number of dims
		self.rotary_dim = head_dim if head_dim % 2 == 0 else head_dim - 1
		if self.rotary_dim <= 0:
			raise ValueError("head_dim too small for RoPE")

		inv_freq = 1.0 / (base ** (arange(0, self.rotary_dim, 2).float() / self.rotary_dim))
		self.register_buffer("inv_freq", inv_freq, persistent=False)

	def _build_sin_cos(self, seq_len: int, device):
		# positions [T]
		pos = arange(seq_len, device=device).float()
		# angles [T, rotary_dim/2]
		freqs = einsum("t,f->tf", pos, self.inv_freq.to(device))
		return sin(freqs), cos(freqs)

	@staticmethod
	def _rotate_half(x: Tensor) -> Tensor:
		# x: [..., rotary_dim]
		x_even = x[..., 0::2]
		x_odd = x[..., 1::2]
		# [-odd, even] interleaved
		out = stack((-x_odd, x_even), dim=-1).flatten(-2)
		return out

	def apply(self, q: Tensor, k: Tensor):
		"""
		q, k: [B, H, T, Hd]
		Apply RoPE to first rotary_dim of the last dimension.
		"""
		B, H, T, Hd = q.shape
		# [T, rotary_dim/2]
		sin, cos = self._build_sin_cos(T, q.device)

		# expand to broadcast: [1,1,T,rotary_dim/2]
		sin = sin[None, None, :, :]
		cos = cos[None, None, :, :]

		rd = self.rotary_dim

		q1, q2 = q[..., :rd], q[..., rd:]
		k1, k2 = k[..., :rd], k[..., rd:]

		# reshape q1/k1 to [..., rotary_dim] where pairs are even/odd already
		# compute: x_rot = x*cos + rotate_half(x)*sin
		# but cos/sin are for pairs and we need them aligned with interleaved dims
		# Expand cos/sin from [.., rd/2] to [.., rd] by repeating each value twice
		# [1,1,T,rd]
		cos2 = cos.repeat_interleave(2, dim=-1)
		sin2 = sin.repeat_interleave(2, dim=-1)

		q1r = (q1 * cos2) + (self._rotate_half(q1) * sin2)
		k1r = (k1 * cos2) + (self._rotate_half(k1) * sin2)

		q_out = cat([q1r, q2], dim=-1)
		k_out = cat([k1r, k2], dim=-1)
		return q_out, k_out
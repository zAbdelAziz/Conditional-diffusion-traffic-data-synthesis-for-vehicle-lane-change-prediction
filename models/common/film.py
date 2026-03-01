from torch import Tensor
from torch.nn import Module, Linear


class FiLM(Module):
	def __init__(self, cond_dim: int, channels: int):
		super().__init__()
		self.to_scale_shift = Linear(cond_dim, 2 * channels)

	def forward(self, cond: Tensor):
		# cond: [B, cond_dim]
		# Project to concatenated scale+shift: [B,2C]
		ss = self.to_scale_shift(cond)
		# Split last dim into two chunks: scale [B,C] and shift [B,C]
		scale, shift = ss.chunk(2, dim=-1)
		return scale, shift

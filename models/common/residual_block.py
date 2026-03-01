from torch import Tensor
from torch.nn import Module, Conv1d, GroupNorm, Dropout
from torch.nn.functional import silu

from models.common import FiLM


class ResBlock1D(Module):
	def __init__(self, channels: int, cond_dim: int, dropout: float = 0.1, groups: int = 8):
		super().__init__()
		# Ensure groups <= channels for GroupNorm
		g = min(groups, channels)
		# First norm over channels: input/output [B,C,T]
		self.gn1 = GroupNorm(g, channels)
		# Conv 1D [BCT]
		self.conv1 = Conv1d(channels, channels, kernel_size=3, padding=1)
		# FiLM produces scale/shift: cond [B,E] to ([B,C],[B,C])
		self.film = FiLM(cond_dim, channels)
		# Normaliztion
		self.gn2 = GroupNorm(g, channels)
		# Dropout
		self.drop = Dropout(dropout)
		# Conv1d [BCT]
		self.conv2 = Conv1d(channels, channels, kernel_size=3, padding=1)

	def forward(self, x: Tensor, cond: Tensor):
		# x: [B,C,T], cond: [B,cond_dim]
		n1 = self.gn1(x)
		n1 = silu(n1)
		h = self.conv1(n1)
		# [B,C], [B,C]
		scale, shift = self.film(cond)
		# Apply FiLM across time by broadcasting: scale/shift -> [B,C,1]
		# h remains [B,C,T]
		# h = (h * (1 + scale)) + shift
		h = h * (1.0 + scale[:, :, None]) + shift[:, :, None]
		n2 = self.gn2(x)
		n2 = silu(n2)
		n2 = self.drop(n2)
		h = self.conv2(n2)
		# add Residual [BCT] + [BCT]
		return x + h
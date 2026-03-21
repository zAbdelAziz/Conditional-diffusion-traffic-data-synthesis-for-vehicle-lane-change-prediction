from torch import Tensor
from torch.nn import Module, Sequential, MultiheadAttention, Linear, SiLU, Dropout, LayerNorm

from models.common import FiLM


class EntityAttentionBlock(Module):
	"""
	Self-attention across the 7 entities at each timestep:
	  ego, LF, LR, RF, RR, F, R
	Input:  [B, T, 7, C]
	Output: [B, T, 7, C]
	"""
	def __init__(self, token_dim: int, cond_dim: int, n_heads: int = 4, dropout: float = 0.1):
		super().__init__()
		assert token_dim % n_heads == 0

		self.ln1 = LayerNorm(token_dim)
		self.attn = MultiheadAttention(embed_dim=token_dim, num_heads=n_heads, dropout=0.0, batch_first=True)
		self.film1 = FiLM(cond_dim, token_dim)

		self.ln2 = LayerNorm(token_dim)
		self.film2 = FiLM(cond_dim, token_dim)
		self.ff = Sequential(Linear(token_dim, token_dim * 4), SiLU(), Dropout(dropout), Linear(token_dim * 4, token_dim))

	def forward(self, x: Tensor, cond: Tensor):
		# x: [B, T, 7, C], cond: [B, E]
		B, T, S, C = x.shape

		h = x.reshape(B * T, S, C)
		c = cond[:, None, :].expand(B, T, cond.size(-1)).reshape(B * T, cond.size(-1))

		y = self.ln1(h)
		scale, shift = self.film1(c)
		y = y * (1.0 + scale[:, None, :]) + shift[:, None, :]
		y, _ = self.attn(y, y, y, need_weights=False)
		h = h + y

		y = self.ln2(h)
		scale, shift = self.film2(c)
		y = y * (1.0 + scale[:, None, :]) + shift[:, None, :]
		h = h + self.ff(y)

		return h.reshape(B, T, S, C)
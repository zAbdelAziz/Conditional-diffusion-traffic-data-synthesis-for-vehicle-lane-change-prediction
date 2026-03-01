from torch import Tensor
from torch.nn import Module, LayerNorm, MultiheadAttention


class SelfAttention1D(Module):
	def __init__(self, channels: int, n_heads: int = 8, dropout: float = 0.0):
		super().__init__()
		# MultiheadAttention requires embed_dim divisible by num_heads
		assert channels % n_heads == 0
		self.n_heads = n_heads
		# LayerNorm normalizes last dim; we apply it on [B,T,C]
		self.ln = LayerNorm(channels)
		# MHA with batch_first=True expects [B,T,C]
		self.mha = MultiheadAttention(embed_dim=channels, num_heads=n_heads, dropout=dropout, batch_first=True)

	def forward(self, x: Tensor):
		# x: [B,C,T]
		# xt: [B,T,C]
		xt = x.transpose(1, 2)
		# Normalize per token embedding: [B,T,C] to [B,T,C]
		h = self.ln(xt)
		# Self-attention: query=key=value=h; output [B,T,C]
		attn_out, _ = self.mha(h, h, h, need_weights=False)
		# Residual connection in token space: [B,T,C]
		out = xt + attn_out
		# Return to channels-first: [B,T,C] -> [B,C,T]
		return out.transpose(1, 2)
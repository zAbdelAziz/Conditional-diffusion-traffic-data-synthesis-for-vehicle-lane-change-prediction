from torch import Tensor, sigmoid
from torch.nn import Module, Sequential, Linear, LayerNorm, Dropout, GELU
from torch.nn.functional import scaled_dot_product_attention


from models.common.positional_encoding import RoPE
from models.common.film import FiLM


class DiffusionTransformerBlock(Module):
	def __init__(self, d_model: int, n_heads: int, cond_dim: int, dropout: float = 0.0, rope_base: float = 10000.0):
		super().__init__()
		assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
		self.d_model = d_model
		self.n_heads = n_heads
		self.head_dim = d_model // n_heads

		self.rope = RoPE(self.head_dim, base=rope_base)

		# Attention projections
		self.qkv = Linear(d_model, 3 * d_model, bias=True)
		self.out = Linear(d_model, d_model, bias=True)

		# Norms
		self.ln1 = LayerNorm(d_model)
		self.ln2 = LayerNorm(d_model)

		# FiLM/AdaLN for each norm
		self.ada1 = FiLM(cond_dim, d_model)
		self.ada2 = FiLM(cond_dim, d_model)

		# Optional dropout
		self.drop = Dropout(dropout)

		# FFN
		self.ff = Sequential(Linear(d_model, 4 * d_model), GELU(), Dropout(dropout), Linear(4 * d_model, d_model))

		# gated residual scaling can help
		self.res_gate1 = Linear(cond_dim, d_model)
		self.res_gate2 = Linear(cond_dim, d_model)

	def _attn(self, x: Tensor):
		# x: [B,T,D]
		B, T, D = x.shape
		# [B,T,3D]
		qkv = self.qkv(x)
		q, k, v = qkv.chunk(3, dim=-1)

		# [B,H,T,Hd]
		q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
		k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
		v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

		# RoPE on q,k
		q, k = self.rope.apply(q, k)

		# scaled dot-product attention
		# [B,H,T,Hd]
		attn = scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.drop.p if self.training else 0.0, is_causal=False)

		# back to [B,T,D]
		attn = attn.transpose(1, 2).contiguous().view(B, T, D)
		return self.out(attn)

	def forward(self, x: Tensor, cond: Tensor):
		# x: [B,T,D]
		# cond: [B,cond_dim]

		# Attention block with AdaLN + gated residual
		h1 = self.ln1(x)
		# [B,D],[B,D]
		s1, b1 = self.ada1(cond)
		h1 = h1 * (1.0 + s1.unsqueeze(1)) + b1.unsqueeze(1)

		a = self._attn(h1)
		# [B,1,D]
		g1 = sigmoid(self.res_gate1(cond)).unsqueeze(1)
		x = x + g1 * self.drop(a)

		# FFN block with AdaLN + gated residual
		h2 = self.ln2(x)
		s2, b2 = self.ada2(cond)
		h2 = h2 * (1.0 + s2.unsqueeze(1)) + b2.unsqueeze(1)

		f = self.ff(h2)
		g2 = sigmoid(self.res_gate2(cond)).unsqueeze(1)
		x = x + g2 * self.drop(f)

		return x


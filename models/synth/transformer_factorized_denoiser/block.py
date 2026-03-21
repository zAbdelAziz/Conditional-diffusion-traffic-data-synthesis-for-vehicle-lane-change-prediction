from torch import Tensor,  sigmoid
from torch.nn import Module, Linear, LayerNorm, Sequential, Dropout, GELU

from torch.nn.functional import scaled_dot_product_attention

from models.common.film import FiLM
from models.common.positional_encoding import RoPE


class FactorizedTransformerBlock(Module):
	"""
	Factorized spatiotemporal transformer block.

	Input/output: [B, T, 7, C]

	Stage 1: entity attention across the 7 slots at each timestep.
	Stage 2: temporal attention across time for each slot independently.

	Each sublayer uses:
	  - LayerNorm
	  - FiLM / AdaLN-style conditioning
	  - gated residual connection
	  - FFN with the same conditioning style
	"""

	def __init__(self, d_model: int, n_heads: int, cond_dim: int, dropout: float = 0.0, rope_base: float = 10000.0, num_slots: int = 7):
		super().__init__()
		assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

		self.d_model = d_model
		self.n_heads = n_heads
		self.head_dim = d_model // n_heads
		self.num_slots = num_slots
		self.drop = Dropout(dropout)

		# RoPE is only used for temporal attention where sequence order is meaningful
		self.time_rope = RoPE(self.head_dim, base=rope_base)

		# Entity attention sublayer
		self.entity_ln1 = LayerNorm(d_model)
		self.entity_ada1 = FiLM(cond_dim, d_model)
		self.entity_qkv = Linear(d_model, 3 * d_model, bias=True)
		self.entity_out = Linear(d_model, d_model, bias=True)
		self.entity_gate1 = Linear(cond_dim, d_model)

		self.entity_ln2 = LayerNorm(d_model)
		self.entity_ada2 = FiLM(cond_dim, d_model)
		self.entity_ff = Sequential(Linear(d_model, 4 * d_model), GELU(), Dropout(dropout), Linear(4 * d_model, d_model))
		self.entity_gate2 = Linear(cond_dim, d_model)

		# Temporal attention sublayer
		self.time_ln1 = LayerNorm(d_model)
		self.time_ada1 = FiLM(cond_dim, d_model)
		self.time_qkv = Linear(d_model, 3 * d_model, bias=True)
		self.time_out = Linear(d_model, d_model, bias=True)
		self.time_gate1 = Linear(cond_dim, d_model)

		self.time_ln2 = LayerNorm(d_model)
		self.time_ada2 = FiLM(cond_dim, d_model)
		self.time_ff = Sequential(Linear(d_model, 4 * d_model), GELU(), Dropout(dropout), Linear(4 * d_model, d_model))
		self.time_gate2 = Linear(cond_dim, d_model)

	def _apply_adaln(self, x: Tensor, cond: Tensor, film: FiLM) -> Tensor:
		# x: [N, L, C], cond: [N, Ccond]
		scale, shift = film(cond)  # [N, C], [N, C]
		return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

	def _mhsa(self, x: Tensor, qkv_proj: Linear, out_proj: Linear, use_rope: bool) -> Tensor:
		# x: [N, L, C]
		n, l, d = x.shape
		qkv = qkv_proj(x)
		q, k, v = qkv.chunk(3, dim=-1)

		# [N, H, L, Hd]
		q = q.view(n, l, self.n_heads, self.head_dim).transpose(1, 2)
		k = k.view(n, l, self.n_heads, self.head_dim).transpose(1, 2)
		v = v.view(n, l, self.n_heads, self.head_dim).transpose(1, 2)

		if use_rope:
			q, k = self.time_rope.apply(q, k)

		attn = scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.drop.p if self.training else 0.0, is_causal=False)
		attn = attn.transpose(1, 2).contiguous().view(n, l, d)
		return out_proj(attn)

	def forward(self, x: Tensor, cond: Tensor) -> Tensor:
		# x: [B, T, S, C], cond: [B, Ccond]
		b, t, s, c = x.shape
		assert s == self.num_slots, f"expected {self.num_slots} slots, got {s}"

		# Entity attention over slots within each timestep
		# [B, T, S, C] -> [B*T, S, C]
		xe = x.reshape(b * t, s, c)
		ce = cond[:, None, :].expand(b, t, cond.size(-1)).reshape(b * t, cond.size(-1))

		h = self.entity_ln1(xe)
		h = self._apply_adaln(h, ce, self.entity_ada1)
		h = self._mhsa(h, self.entity_qkv, self.entity_out, use_rope=False)
		# [B*T, 1, C]
		g = sigmoid(self.entity_gate1(ce)).unsqueeze(1)
		xe = xe + g * self.drop(h)

		h = self.entity_ln2(xe)
		h = self._apply_adaln(h, ce, self.entity_ada2)
		h = self.entity_ff(h)
		g = sigmoid(self.entity_gate2(ce)).unsqueeze(1)
		xe = xe + g * self.drop(h)

		x = xe.reshape(b, t, s, c)

		# Temporal attention over time within each slot
		# [B, T, S, C] -> [B*S, T, C]
		xt = x.transpose(1, 2).reshape(b * s, t, c)
		ct = cond[:, None, :].expand(b, s, cond.size(-1)).reshape(b * s, cond.size(-1))

		h = self.time_ln1(xt)
		h = self._apply_adaln(h, ct, self.time_ada1)
		h = self._mhsa(h, self.time_qkv, self.time_out, use_rope=True)
		# [B*S, 1, C]
		g = sigmoid(self.time_gate1(ct)).unsqueeze(1)
		xt = xt + g * self.drop(h)

		h = self.time_ln2(xt)
		h = self._apply_adaln(h, ct, self.time_ada2)
		h = self.time_ff(h)
		g = sigmoid(self.time_gate2(ct)).unsqueeze(1)
		xt = xt + g * self.drop(h)

		x = xt.reshape(b, s, t, c).transpose(1, 2).contiguous()
		return x
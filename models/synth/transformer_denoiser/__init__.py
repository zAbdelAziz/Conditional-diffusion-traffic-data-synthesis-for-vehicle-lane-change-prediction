from typing import Optional

from torch import Tensor, full, full_like, rand, where, long
from torch.nn import ModuleList, Module, Linear, LayerNorm,  Embedding, Sequential, SiLU

from models.common.time_embed import SinusoidalTimeEmbedding
from models.synth.transformer_denoiser.block import DiffusionTransformerBlock


class TransformerDenoiserModel(Module):
	def __init__(self, data_dim: int, d_model: int = 256, n_heads: int = 8, n_layers: int = 6, dropout: float = 0.0,
			num_classes: Optional[int] = 3, cfg_drop_prob: float = 0.15, rope_base: float = 10000.0):
		super().__init__()
		self.data_dim = data_dim
		self.d_model = d_model
		self.num_classes = num_classes
		self.cfg_drop_prob = cfg_drop_prob

		self.in_proj = Linear(data_dim, d_model)
		self.out_proj = Linear(d_model, data_dim)

		# time embedding To cond
		self.time_emb = SinusoidalTimeEmbedding(d_model)
		self.time_mlp = Sequential(Linear(d_model, d_model), SiLU(), Linear(d_model, d_model))

		# class embedding with null token for CFG
		if num_classes is not None:
			# last index
			self.null_class = num_classes
			self.y_emb = Embedding(num_classes + 1, d_model)
		else:
			self.null_class = None
			self.y_emb = None

		# optional cond refinement (helps)
		self.cond_mlp = Sequential(Linear(d_model, d_model), SiLU(), Linear(d_model, d_model))

		self.blocks = ModuleList([
			DiffusionTransformerBlock(d_model=d_model, n_heads=n_heads, cond_dim=d_model, dropout=dropout, rope_base=rope_base)
			for _ in range(n_layers)
		])

		self.final_ln = LayerNorm(d_model)

	def _make_cfg_labels(self, t: Tensor, y: Optional[Tensor], train: bool) -> Optional[Tensor]:
		if self.y_emb is None:
			return None

		B = t.shape[0]
		null = self.null_class

		if y is None:
			y_idx = full((B,), null, device=t.device, dtype=long)
			return y_idx

		y_idx = y.to(device=t.device).long().view(-1)

		# harden invalid labels -> null
		valid = (y_idx >= 0) & (y_idx < self.num_classes)
		y_idx = where(valid, y_idx, full_like(y_idx, null))

		# CFG drop during training
		if train and self.cfg_drop_prob > 0.0:
			drop = rand(B, device=t.device) <= self.cfg_drop_prob
			y_idx = y_idx.clone()
			y_idx[drop] = null

		return y_idx

	def forward(self, x_t: Tensor, t: Tensor, y: Optional[Tensor] = None) -> Tensor:
		# x_t: [B,T,D]
		# [B,T,d_model]
		x = self.in_proj(x_t)

		# time conditioning
		# [B,d_model]
		temb = self.time_mlp(self.time_emb(t))

		# class conditioning + CFG drop
		if self.y_emb is not None:
			y_idx = self._make_cfg_labels(t, y, train=self.training)
			# [B,d_model]
			yemb = self.y_emb(y_idx)
			cond = temb + yemb
		else:
			cond = temb
		# [B,d_model]
		cond = self.cond_mlp(cond)

		# transformer blocks
		for blk in self.blocks:
			x = blk(x, cond)

		x = self.final_ln(x)
		# [B,T,D]
		eps_hat = self.out_proj(x)
		return eps_hat
from typing import Optional

from torch import Tensor, arange, cat, full, full_like, long, rand, where, sigmoid
from torch.nn import Module, ModuleList, Linear, LayerNorm, Embedding, Sequential, SiLU, Dropout, GELU

from models.common.time_embed import SinusoidalTimeEmbedding
from models.synth.transformer_factorized_denoiser.block import FactorizedTransformerBlock

class FactorizedTransformerDenoiserModel(Module):
	"""
	Structured diffusion transformer with:
	  - classifier-free guidance conditioning
	  - ego + 6-neighbor tokenization
	  - slot embeddings
	  - factorized entity/time attention blocks
	  - structured outputs matching the hybrid U-Net:
		  eps_ego   : [B, T, 2]
		  eps_slots : [B, T, 6, 2]
		  p_logits  : [B, T, 6]
	"""

	def __init__(self, token_dim: int = 256, n_heads: int = 8, n_layers: int = 8, dropout: float = 0.0,
				 num_classes: Optional[int] = 3, cfg_drop_prob: float = 0.15, rope_base: float = 10000.0, num_slots: int = 7):
		super().__init__()
		assert num_slots == 7, "this model currently assumes 1 ego + 6 neighbor slots"

		self.data_dim = 20
		self.token_dim = token_dim
		self.num_slots = num_slots
		self.num_classes = num_classes
		self.cfg_drop_prob = cfg_drop_prob

		# Structured tokenization
		self.ego_proj = Linear(2, token_dim)
		self.nbr_proj = Linear(3, token_dim)
		self.slot_emb = Embedding(num_slots, token_dim)
		self.token_norm = LayerNorm(token_dim)

		# Time embedding -> condition
		self.time_emb = SinusoidalTimeEmbedding(token_dim)
		self.time_mlp = Sequential(Linear(token_dim, token_dim), SiLU(), Linear(token_dim, token_dim))

		# Class embedding with null token for CFG
		if num_classes is not None:
			self.null_class = num_classes
			self.y_emb = Embedding(num_classes + 1, token_dim)
		else:
			self.null_class = None
			self.y_emb = None

		# Refine condition before injecting into blocks
		self.cond_mlp = Sequential(Linear(token_dim, token_dim), SiLU(), Linear(token_dim, token_dim))

		self.blocks = ModuleList([FactorizedTransformerBlock(d_model=token_dim, n_heads=n_heads, cond_dim=token_dim, dropout=dropout,
															 rope_base=rope_base, num_slots=num_slots)
								  for _ in range(n_layers)])

		self.final_ln = LayerNorm(token_dim)

		# Structured output heads
		self.ego_eps_head = Linear(token_dim, 2)
		self.nbr_eps_head = Linear(token_dim, 2)
		self.nbr_mask_head = Linear(token_dim, 1)

	def _make_cfg_labels(self, t: Tensor, y: Optional[Tensor], train: bool) -> Optional[Tensor]:
		if self.y_emb is None:
			return None

		b = t.shape[0]
		null = self.null_class

		if y is None:
			return full((b,), null, device=t.device, dtype=long)

		y_idx = y.to(device=t.device).long().view(-1)

		valid = (y_idx >= 0) & (y_idx < self.num_classes)
		y_idx = where(valid, y_idx, full_like(y_idx, null))

		if train and self.cfg_drop_prob > 0.0:
			drop = rand(b, device=t.device) <= self.cfg_drop_prob
			y_idx = y_idx.clone()
			y_idx[drop] = null

		return y_idx

	def _make_cond(self, t: Tensor, y: Optional[Tensor]) -> Tensor:
		temb = self.time_mlp(self.time_emb(t))  # [B, C]

		if self.y_emb is not None:
			y_idx = self._make_cfg_labels(t, y, train=self.training)
			yemb = self.y_emb(y_idx)
			cond = temb + yemb
		else:
			cond = temb

		return self.cond_mlp(cond)

	def _tokenize(self, x_t: Tensor) -> Tensor:
		# x_t: [B, T, 20]
		b, t, d = x_t.shape
		assert d == 20, f"expected D=20, got {d}"

		# [B, T, 2]
		ego = x_t[:, :, 0:2]
		# [B, T, 6, 3]
		nbr = x_t[:, :, 2:20].reshape(b, t, 6, 3)

		# [B, T, 1, C]
		ego_tok = self.ego_proj(ego).unsqueeze(2)
		# [B, T, 6, C]
		nbr_tok = self.nbr_proj(nbr)

		# [B, T, 7, C]
		x = cat([ego_tok, nbr_tok], dim=2)

		slot_ids = arange(self.num_slots, device=x_t.device, dtype=long)
		x = x + self.slot_emb(slot_ids)[None, None, :, :]
		x = self.token_norm(x)
		return x

	def forward(self, x_t: Tensor, t: Tensor, y: Optional[Tensor] = None):
		# x_t: [B, T, 20]
		# t : [B]
		# y : [B] or None

		# [B, C]
		cond = self._make_cond(t, y)
		# [B, T, 7, C]
		x = self._tokenize(x_t)

		for blk in self.blocks:
			x = blk(x, cond)

		x = self.final_ln(x)

		# [B, T, C]
		ego_feat = x[:, :, 0, :]
		# [B, T, 6, C]
		nbr_feat = x[:, :, 1:, :]

		# [B, T, 2]
		eps_ego = self.ego_eps_head(ego_feat)
		# [B, T, 6, 2]
		eps_slots = self.nbr_eps_head(nbr_feat)
		# [B, T, 6]
		p_logits = self.nbr_mask_head(nbr_feat).squeeze(-1)

		return {"eps_ego": eps_ego, "eps_slots": eps_slots, "p_logits": p_logits}

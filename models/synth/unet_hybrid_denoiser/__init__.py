from typing import Optional

from torch import Tensor, arange, cat, full, full_like, rand, where, long
from torch.nn import Module, ModuleList, Linear, Conv1d, Embedding, GroupNorm, LayerNorm

from torch.nn.functional import pad, silu

from models.common import SinusoidalTimeEmbedding, CondMLP, Upsample1D, Downsample1D, SelfAttention1D, ResBlock1D
from models.synth.unet_hybrid_denoiser.entity_attn import EntityAttentionBlock


class HybridUNetDenoiserModel(Module):
	def __init__(self, token_dim: int = 32, base_channels: int = 128, cond_dim: int = 256, entity_heads: int = 4, attn_heads: int = 8,
				 n_entity_blocks: int = 2, use_attn_mid: bool = True, use_attn_low: bool = True, num_classes: int = 3,
				 dropout: float = 0.1, cfg_drop_prob: float = 0.15):
		super().__init__()

		self.token_dim = token_dim
		self.cond_dim = cond_dim
		self.num_classes = num_classes
		self.cfg_drop_prob = cfg_drop_prob

		# Conditioning
		self.t_embed = SinusoidalTimeEmbedding(cond_dim)
		self.y_embed = Embedding(num_classes + 1, cond_dim)   # last row = null label
		self.cond_mlp = CondMLP(cond_dim, cond_dim, hidden_dim=512)

		# Tokenization
		self.ego_proj = Linear(2, token_dim)
		self.nbr_proj = Linear(3, token_dim)
		self.slot_embed = Embedding(7, token_dim)
		self.token_norm = LayerNorm(token_dim)

		self.entity_blocks = ModuleList([EntityAttentionBlock(token_dim, cond_dim, n_heads=entity_heads, dropout=dropout)
										 for _ in range(n_entity_blocks)])

		# Temporal U-Net over flattened entity tokens
		in_channels = 7 * token_dim
		self.in_proj = Conv1d(in_channels, base_channels, kernel_size=3, padding=1)
		self.out_norm = GroupNorm(8, base_channels)
		self.out_proj = Conv1d(base_channels, in_channels, kernel_size=3, padding=1)

		c2 = base_channels * 2
		c4 = base_channels * 4

		self.rb1 = ResBlock1D(base_channels, cond_dim, dropout)
		self.rb2 = ResBlock1D(base_channels, cond_dim, dropout)
		self.down1 = Downsample1D(base_channels, c2)

		self.rb3 = ResBlock1D(c2, cond_dim, dropout)
		self.rb4 = ResBlock1D(c2, cond_dim, dropout)
		self.attn_low = SelfAttention1D(c2, n_heads=attn_heads, dropout=0.0) if use_attn_low else None
		self.down2 = Downsample1D(c2, c4)

		self.rb5 = ResBlock1D(c4, cond_dim, dropout)
		self.attn_mid = SelfAttention1D(c4, n_heads=attn_heads, dropout=0.0) if use_attn_mid else None
		self.rb6 = ResBlock1D(c4, cond_dim, dropout)

		self.up1 = Upsample1D(c4, c2)
		self.rb7 = ResBlock1D(c2, cond_dim, dropout)
		self.rb8 = ResBlock1D(c2, cond_dim, dropout)

		self.up2 = Upsample1D(c2, base_channels)
		self.rb9 = ResBlock1D(base_channels, cond_dim, dropout)
		self.rb10 = ResBlock1D(base_channels, cond_dim, dropout)

		# Output heads
		self.ego_eps_head = Linear(token_dim, 2)
		self.nbr_eps_head = Linear(token_dim, 2)
		self.nbr_mask_head = Linear(token_dim, 1)

	def forward(self, x_t: Tensor, t: Tensor, y: Optional[Tensor] = None):
		"""
		Output	"eps_ego": [B, T, 2]
				"eps_slots": [B, T, 6, 2]
				"p_logits":  [B, T, 6]}
		"""
		B, T, D = x_t.shape
		assert D == 20, f"expected D=20, got {D}"

		cond = self._make_cond(t, y, train=self.training)

		# Tokenize raw state into 7 entity tokens
		# [B,T,2]
		ego = x_t[:, :, 0:2]
		# [B,T,6,3]
		nbr = x_t[:, :, 2:20].reshape(B, T, 6, 3)

		# [B,T,1,C]
		ego_tok = self.ego_proj(ego).unsqueeze(2)
		# [B,T,6,C]
		nbr_tok = self.nbr_proj(nbr)
		# [B,T,7,C]
		tok = cat([ego_tok, nbr_tok], dim=2)

		slot_ids = arange(7, device=x_t.device, dtype=long)
		tok = tok + self.slot_embed(slot_ids)[None, None, :, :]
		tok = self.token_norm(tok)

		for blk in self.entity_blocks:
			tok = blk(tok, cond)

		# Flatten entities into channels then temporal U-Net
		# [B, 7C, T]
		x = tok.reshape(B, T, 7 * self.token_dim).transpose(1, 2)
		x0 = self.in_proj(x)

		h1 = self.rb2(self.rb1(x0, cond), cond)
		d1 = self.down1(h1)

		h2 = self.rb4(self.rb3(d1, cond), cond)
		if self.attn_low is not None:
			h2 = self.attn_low(h2)
		d2 = self.down2(h2)

		m = self.rb5(d2, cond)
		if self.attn_mid is not None:
			m = self.attn_mid(m)
		m = self.rb6(m, cond)

		u1 = self.up1(m)
		u1 = self._match_time(u1, h2)
		u1 = u1 + h2
		u1 = self.rb8(self.rb7(u1, cond), cond)

		u2 = self.up2(u1)
		u2 = self._match_time(u2, h1)
		u2 = u2 + h1
		u2 = self.rb10(self.rb9(u2, cond), cond)

		tok_delta = self.out_proj(silu(self.out_norm(u2)))   # [B,7C,T]
		tok_delta = tok_delta.transpose(1, 2).reshape(B, T, 7, self.token_dim)

		tok_out = tok + tok_delta

		# [B,T,C]
		ego_feat = tok_out[:, :, 0, :]
		# [B,T,6,C]
		nbr_feat = tok_out[:, :, 1:, :]

		# [B,T,2]
		eps_ego = self.ego_eps_head(ego_feat)
		# [B,T,6,2]
		eps_slots = self.nbr_eps_head(nbr_feat)
		# [B,T,6]
		p_logits = self.nbr_mask_head(nbr_feat).squeeze(-1)

		return {"eps_ego": eps_ego, "eps_slots": eps_slots, "p_logits": p_logits}

	def _match_time(self, x: Tensor, ref: Tensor):
		tx, tr = x.size(-1), ref.size(-1)
		if tx == tr:
			return x
		if tx > tr:
			return x[..., :tr]
		return pad(x, (0, tr - tx))

	def _make_cond(self, t: Tensor, y: Optional[Tensor], train: bool):
		B = t.shape[0]
		te = self.t_embed(t)

		null = self.num_classes
		if y is None:
			y_idx = full((B,), null, device=t.device, dtype=long)
		else:
			y_idx = y.to(device=t.device).long().view(-1)
			valid = (y_idx >= 0) & (y_idx < self.num_classes)
			y_idx = where(valid, y_idx, full_like(y_idx, null))

			if train and self.cfg_drop_prob > 0.0:
				drop = rand(B, device=t.device) <= self.cfg_drop_prob
				y_idx = y_idx.clone()
				y_idx[drop] = null

		ye = self.y_embed(y_idx)
		return self.cond_mlp(te + ye)
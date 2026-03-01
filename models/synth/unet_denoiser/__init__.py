from typing import Optional

from torch import Tensor, full, full_like, where, rand, long
from torch.nn import Module, Conv1d, Embedding, GroupNorm
from torch.nn.functional import silu, pad

from models.common import SinusoidalTimeEmbedding, CondMLP, Upsample1D, Downsample1D, SelfAttention1D, ResBlock1D


class UNETDenoiserModel(Module):
    def __init__(self, data_dim: int = 33, base_channels: int = 128, cond_dim: int = 256,
                 attn_heads: int = 8, use_attn_mid: bool = True, use_attn_low: bool = True,
                 num_classes: int = 3, dropout: float = 0.1, cfg_drop_prob: float = 0.15,):
        super().__init__()

        self.data_dim = data_dim
        self.cond_dim = cond_dim
        self.num_classes = num_classes
        self.cfg_drop_prob = cfg_drop_prob

        # Embeddings
        # Time Embedding
        # [t] to [B, E]
        self.t_embed = SinusoidalTimeEmbedding(cond_dim)
        # Y Embedding [with null class] [B,E]
        self.y_embed = Embedding(num_classes + 1, cond_dim)
        # Conditional MLP [B,E]
        self.cond_mlp = CondMLP(cond_dim, cond_dim, hidden_dim=512)

        # Input/Output Projection
        # [B,D,T] to [B, base_channels, T]
        self.in_proj = Conv1d(data_dim, base_channels, kernel_size=3, padding=1)
        # [B, base_channels, T]
        self.out_norm = GroupNorm(min(8, base_channels), base_channels)
        # [B, base_channels, T] to [B,D,T]
        self.out_proj = Conv1d(base_channels, data_dim, kernel_size=3, padding=1)

        base_channels_2 = base_channels * 2
        base_channels_4 = base_channels * 4

        # Down Path
        # [B,base_channels,T]
        self.rb1 = ResBlock1D(base_channels, cond_dim, dropout)
        # [B,base_channels,T]
        self.rb2 = ResBlock1D(base_channels, cond_dim, dropout)
        # [B,base_channels,T] TO [B,base_channels_2,T/2]
        self.down1 = Downsample1D(base_channels, base_channels_2)

        # [B,base_channels_2,T/2]
        self.rb3 = ResBlock1D(base_channels_2, cond_dim, dropout)
        # [B,base_channels_2,T/2]
        self.rb4 = ResBlock1D(base_channels_2, cond_dim, dropout)
        # Self Attention [B,base_channels_2,T/2]
        self.attn_low = SelfAttention1D(base_channels_2, n_heads=attn_heads, dropout=0.0) if use_attn_low else None
        # [B,base_channels_2,T/2] TO [B,base_channels_4,T/4]
        self.down2 = Downsample1D(base_channels_2, base_channels_4)

        # Core / Mid Path
        # [B,base_channels_4,T/4]
        self.rb5 = ResBlock1D(base_channels_4, cond_dim, dropout)
        # [B,base_channels_4,T/4]
        self.attn_mid = SelfAttention1D(base_channels_4, n_heads=attn_heads, dropout=0.0) if use_attn_mid else None
        # [B,base_channels_4,T/4]
        self.rb6 = ResBlock1D(base_channels_4, cond_dim, dropout)

        # [B,base_channels_4,T/4] TO [B,base_channels_2,~T/2]
        self.up1 = Upsample1D(base_channels_4, base_channels_2)
        # [B,base_channels_2,T/2]
        self.rb7 = ResBlock1D(base_channels_2, cond_dim, dropout)
        # [B,base_channels_2,T/2]
        self.rb8 = ResBlock1D(base_channels_2, cond_dim, dropout)

        # [B,base_channels_2,T/2] TO [B,base_channels,~T]
        self.up2 = Upsample1D(base_channels_2, base_channels)
        # [B,base_channels,T]
        self.rb9 = ResBlock1D(base_channels, cond_dim, dropout)
        # [B,base_channels,T]
        self.rb10 = ResBlock1D(base_channels, cond_dim, dropout)

    def forward(self, x_t: Tensor, t: Tensor, y: Tensor = None):
        # x_t: [B,T,D] and conv expects [B,D,T]
        x = x_t.transpose(1, 2)

        # Conditional
        cond = self._make_cond(t, y, train=self.training)

        # [B, base_channels, T]
        x0 = self.in_proj(x)

        # Down Path 1
        # [B, base_channels, T]
        rb1 = self.rb1(x0, cond)
        # Skip connection [B, base_channels, T]
        h1 = self.rb2(rb1, cond)
        # [B, base_channels * 2, T/2]
        d1 = self.down1(h1)

        # Down Path 2
        # [B, base_channels * 2, T/2]
        rb3 = self.rb3(d1, cond)
        # Skip connection [B, base_channels * 2, T/2]
        h2 = self.rb4(rb3, cond)  # [B,C2,T/2]
        if self.attn_low is not None:
            h2 = self.attn_low(h2)
        # [B, base_channels * 4, T/4]
        d2 = self.down2(h2)

        # Core
        m = self.rb5(d2, cond)
        if self.attn_mid is not None:
            m = self.attn_mid(m)
        m = self.rb6(m, cond)

        # Up Path 1
        u1 = self.up1(m)
        # Crop/pad time dimension to match skip h2 time length exactly
        u1 = self._match_time(u1, h2)
        # Add skip connection
        u1 = u1 + h2
        rb7 = self.rb7(u1, cond)
        u1 = self.rb8(rb7, cond)

        # Up Path 2
        u2 = self.up2(u1)
        # Crop/pad time dimension to match skip h1 time length exactly (which is original T)
        u2 = self._match_time(u2, h1)
        # Add skip connection
        u2 = u2 + h1
        u2 = self.rb10(self.rb9(u2, cond), cond)

        # [B,D,T]
        eps = self.out_proj(silu(self.out_norm(u2)))
        # [B,T,D]
        return eps.transpose(1, 2)

    def _match_time(self, x: Tensor, ref: Tensor):
        # Utility to force x time dimension to match reference ref

        # x: [B,base_channels,T]
        # ref: [B,base_channels,Tref]
        tx, tr = x.size(-1), ref.size(-1)
        # If already equal, do nothing
        if tx == tr:
            return x
        # If x is longer crop end: [B,C,Tx] TO [B,C,Tr]
        if tx > tr:
            return x[..., :tr]
        # If x is shorter pad zeros at end to Tr
        return pad(x, (0, tr - tx))

    def _make_cond(self, t: Tensor, y: Optional[Tensor], train: bool):
        # Build conditioning vector cond from time t and optional label y with CFG drop

        B = t.shape[0]
        # Time embedding: [B] TO [B,cond_dim]
        te = self.t_embed(t)

        # Null class index is the last embedding row
        null = self.num_classes  # last embedding index is "null"

        # If no labels provided, set all to null: y_idx [B]
        if y is None:
            y_idx = full((B,), null, device=t.device, dtype=long)
        else:
            # Force labels to int64 and flatten to [B]
            y_idx = y.to(device=t.device).long().view(-1)

            # HARDENING: Mark valid labels in [0, num_classes-1] and invalid become null
            valid = (y_idx >= 0) & (y_idx < self.num_classes)
            # where keeps valid label else inserts null; shape stays [B]
            y_idx = where(valid, y_idx, full_like(y_idx, null))

            # Classifier-free guidance drop: randomly set some labels to null during training
            if train and self.cfg_drop_prob > 0:
                # drop mask: [B] boolean
                drop = rand(B, device=t.device) <= self.cfg_drop_prob
                # Clone then set dropped positions to null
                y_idx = y_idx.clone()
                y_idx[drop] = null

        # Label embedding lookup: y_idx [B] to ye [B,E]
        ye = self.y_embed(y_idx)
        # Combine time + label then MLP: [B,E] TO [B,E].
        cond = self.cond_mlp(te + ye)
        # Final conditioning vector used by FiLM in all ResBlocks
        return cond
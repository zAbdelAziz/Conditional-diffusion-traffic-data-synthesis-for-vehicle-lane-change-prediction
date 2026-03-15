from torch.nn import Module, Sequential, Linear, LayerNorm, GELU, Dropout, TransformerEncoderLayer, TransformerEncoder

from models.common.add_attn import AdditiveAttention
from models.common.positional_encoding import PositionalEncoding
from models.downstream.transformer_bigru_classifier.stacked_bigru import StackedBiGRU


class TransformerBiGRUClassifierModel(Module):
	"""
	Transformer + Stacked BiGRU + Additive Attention classifier
	"""
	def __init__(self, input_dim: int, d_model: int = 128, n_heads: int = 8, n_tf_layers: int = 2, dropout: float = 0.1,
				 n_bigru_layers: int = 1, num_classes: int = 3, max_len: int = 256, bigru_hidden_sizes=(256, 128, 64, 32),
				 positional: bool = True):

		super().__init__()
		self.in_proj = Linear(input_dim, d_model)
		self.positional = positional
		self.pos = PositionalEncoding(d_model=d_model, max_len=max_len)

		enc_layer = TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, dropout=dropout,
											batch_first=True, activation="gelu", norm_first=True)
		self.transformer = TransformerEncoder(enc_layer, num_layers=n_tf_layers)

		self.bigru = StackedBiGRU(input_dim=d_model, hidden_sizes=list(bigru_hidden_sizes), num_layers=n_bigru_layers, dropout=dropout)

		self.attn = AdditiveAttention(self.bigru.out_dim)

		self.head = Sequential(LayerNorm(self.bigru.out_dim), Linear(self.bigru.out_dim, 128), GELU(),
							   Dropout(dropout), Linear(128, num_classes))

	def forward(self, x):
		# x: [B,T,D]
		# [B,T,d_model]
		z = self.in_proj(x)
		if self.positional:
			# inject order
			z = self.pos(z)
		# [B,T,d_model]
		z = self.transformer(z)
		# [B,T,2*h_last]
		z = self.bigru(z)
		# [B,2*h_last]
		ctx, _ = self.attn(z)
		# [B,3]
		return self.head(ctx)

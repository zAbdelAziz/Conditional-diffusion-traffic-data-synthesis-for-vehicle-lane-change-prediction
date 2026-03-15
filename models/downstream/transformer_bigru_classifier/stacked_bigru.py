from torch.nn import Module, ModuleList, GRU, Dropout


class StackedBiGRU(Module):
	"""
	BiGRU with varying hidden sizes per layer
	Each layer is bidirectional so output dim is 2*hidden
	"""
	def __init__(self, input_dim: int, hidden_sizes, num_layers: int = 1,  dropout: float = 0.1):
		super().__init__()
		self.layers = ModuleList()
		d_in = input_dim
		for i, h in enumerate(hidden_sizes):
			self.layers.append(GRU(input_size=d_in, hidden_size=h, num_layers=num_layers, batch_first=True, bidirectional=True))
			d_in = 2 * h
		self.dropout = Dropout(dropout)

	@property
	def out_dim(self):
		# last layer output dim
		last = self.layers[-1].hidden_size
		return 2 * last

	def forward(self, x):
		out = x
		for gru in self.layers:
			# [B,T,2H]
			out, _ = gru(out)
			out = self.dropout(out)
		return out
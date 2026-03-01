from torch import Tensor
from torch.nn import Module, Sequential, Linear, SiLU


class CondMLP(Module):
	def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512):
		super().__init__()
		# Sequential MLP:
		# 	Linear: [B,in_dim] to [B,hidden_dim]
		# 	SiLU: nonlinearity
		# 	Linear: [B,hidden_dim] to [B,out_dim]
		self.net = Sequential(Linear(in_dim, hidden_dim), SiLU(), Linear(hidden_dim, out_dim))

	def forward(self, x: Tensor):
		return self.net(x)
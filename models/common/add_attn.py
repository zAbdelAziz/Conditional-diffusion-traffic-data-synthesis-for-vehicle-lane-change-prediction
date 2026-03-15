from torch import softmax
from torch.nn import Module, Sequential, Linear, Tanh


class AdditiveAttention(Module):
	def __init__(self, dim: int):
		super().__init__()
		self.proj = Sequential(Linear(dim, dim), Tanh(), Linear(dim, 1))

	def forward(self, x):
		# x: [B,T,D]
		# [B,T]
		scores = self.proj(x).squeeze(-1)
		# [B,T,1]
		w = softmax(scores, dim=1).unsqueeze(-1)
		# [B,D]
		ctx = (x * w).sum(dim=1)
		return ctx, w

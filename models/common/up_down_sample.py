from torch import Tensor
from torch.nn import Module, Conv1d
from torch.nn.functional import interpolate


class Upsample1D(Module):
	def __init__(self, c_in: int, c_out: int):
		super().__init__()
		self.conv = Conv1d(c_in, c_out, kernel_size=3, padding=1)

	def forward(self, x: Tensor):
		# nearest upsample by 2 then conv
		x = interpolate(x, scale_factor=2.0, mode="nearest")
		return self.conv(x)


class Downsample1D(Module):
	def __init__(self, c_in: int, c_out: int):
		super().__init__()
		self.conv = Conv1d(c_in, c_out, kernel_size=4, stride=2, padding=1)

	def forward(self, x: Tensor):
		return self.conv(x)
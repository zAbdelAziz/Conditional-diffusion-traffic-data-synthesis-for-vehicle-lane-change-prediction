from torch import zeros_like
from torch.nn import Module, Conv1d


class UNETDenoiserModel(Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.in_proj = Conv1d(14, 128, kernel_size=3, padding=1)

    def forward(self, x, t, y):
        return zeros_like(x)
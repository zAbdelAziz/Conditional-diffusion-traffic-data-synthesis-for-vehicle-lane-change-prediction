from torch.nn import Module

class GaussianDiffusionModel(Module):
    def __init__(self, **kwargs):
        super().__init__()
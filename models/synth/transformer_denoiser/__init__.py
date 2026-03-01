from torch.nn import Module


class TransformerDenoiserModel(Module):
    def __init__(self, **kwargs):
        super().__init__()
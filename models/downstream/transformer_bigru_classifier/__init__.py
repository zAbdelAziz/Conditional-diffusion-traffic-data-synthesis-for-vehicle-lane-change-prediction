from torch.nn import Module


class TransformerBiGRUClassifierModel(Module):
    def __init__(self, **kwargs):
        super().__init__()
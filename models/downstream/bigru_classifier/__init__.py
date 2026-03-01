from torch.nn import Module


class BiGRUClassifierModel(Module):
    def __init__(self, **kwargs):
        super().__init__()
from torch.nn import Module


class XLSTMClassifierModel(Module):
    def __init__(self, **kwargs):
        super().__init__()
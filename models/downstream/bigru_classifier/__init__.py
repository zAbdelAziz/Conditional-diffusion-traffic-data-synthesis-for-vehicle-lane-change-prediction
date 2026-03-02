from torch.nn import Module, Sequential, LayerNorm, Linear, GRU


class BiGRUClassifierModel(Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.gru = GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=True, dropout=dropout,
                       batch_first=True)

        self.head = Sequential(LayerNorm(hidden_dim * 2), Linear(hidden_dim * 2, num_classes))

    def forward(self, x):
        # out: [B, T, 2H]
        out, h = self.gru(x)
        # [B, 2H]
        last = out[:, -1, :]
        # [B, 3]
        logits = self.head(last)
        return logits
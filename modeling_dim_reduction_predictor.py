from torch import nn
class dim_reductionPredictor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=None):
        super().__init__()

        self.hidden_size = hidden_size if hidden_size else input_size

        self.dropout = nn.Dropout(0.1)
        self.predictor = nn.Sequential(
            nn.LayerNorm(input_size),
            self.dropout,
            nn.Linear(input_size, output_size),
            # nn.Linear(input_size, self.hidden_size),
            # nn.Linear(self.hidden_size, self.hidden_size),
            # nn.GELU(),
            # nn.Linear(self.hidden_size, self.hidden_size),
            # nn.Linear(self.hidden_size, self.hidden_size),
            # nn.GELU(),
            # nn.Linear(self.hidden_size, output_size),
        )

    def forward(self, hidden_states):
        return self.predictor(hidden_states)
import torch.nn as nn

class WindSpeedPredictionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,num_layers=num_layers, batch_first=True)
        self.activation = nn.ReLU()
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.activation(x[:, -1, :])
        x = self.dropout(x)
        x = self.fc(x)
        return x
    

class HumidityMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(HumidityMLP, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.hidden_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout / 1.5),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 4),
            nn.Dropout(dropout / 2)
        )

        self.output_layer = nn.Linear(hidden_dim // 4, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        out = self.output_layer(x)
        return out
import math

from .dependencies import require_torch


torch = require_torch()
nn = torch.nn
F = torch.nn.functional


class BaselineMLP(nn.Module):
    def __init__(self, input_dim: int = 27) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=0.01, nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.fc1(x)), 0.01)
        out = self.dropout1(out)
        out = F.leaky_relu(self.bn2(self.fc2(out)), 0.01)
        out = self.dropout2(out)
        return torch.sigmoid(self.fc3(out))


class TemporalCNN(nn.Module):
    def __init__(self, input_dim: int = 27, n_filters: int = 16) -> None:
        super().__init__()
        self.conv3 = nn.Conv1d(1, n_filters, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(1, n_filters, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(1, n_filters, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(n_filters * 3)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(n_filters * 3, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_uniform_(module.weight, a=0.01, nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        x = x.unsqueeze(1)
        c3 = F.leaky_relu(self.conv3(x), 0.01)
        c5 = F.leaky_relu(self.conv5(x), 0.01)
        c7 = F.leaky_relu(self.conv7(x), 0.01)
        out = torch.cat([c3, c5, c7], dim=1)
        out = self.bn(out)
        out = out.mean(dim=2)
        out = self.dropout(out)
        return torch.sigmoid(self.fc(out))


class MultiScaleCNN(nn.Module):
    def __init__(self, input_dim: int = 27, n_filters: int = 16) -> None:
        super().__init__()
        self.conv3_1 = nn.Conv1d(1, n_filters, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv1d(1, n_filters, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(n_filters * 2)

        self.conv3_2 = nn.Conv1d(n_filters * 2, n_filters * 2, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv1d(n_filters * 2, n_filters * 2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(n_filters * 4)

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(n_filters * 4, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_uniform_(module.weight, a=0.01, nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        x = x.unsqueeze(1)
        c3 = F.leaky_relu(self.conv3_1(x), 0.01)
        c5 = F.leaky_relu(self.conv5_1(x), 0.01)
        out = torch.cat([c3, c5], dim=1)
        out = self.bn1(out)

        c3 = F.leaky_relu(self.conv3_2(out), 0.01)
        c5 = F.leaky_relu(self.conv5_2(out), 0.01)
        out = torch.cat([c3, c5], dim=1)
        out = self.bn2(out)

        out = out.mean(dim=2)
        out = self.dropout(out)
        return torch.sigmoid(self.fc(out))


class GRUEncoder(nn.Module):
    def __init__(self, input_dim: int = 27, hidden_size: int = 32, num_layers: int = 2) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        x = x.unsqueeze(2)
        _, hidden = self.gru(x)
        h_fwd = hidden[-2]
        h_bwd = hidden[-1]
        out = torch.cat([h_fwd, h_bwd], dim=1)
        out = self.dropout(out)
        return torch.sigmoid(self.fc(out))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 50) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SmallTransformer(nn.Module):
    def __init__(self, input_dim: int = 27, d_model: int = 32, nhead: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=0.3,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.embed(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return torch.sigmoid(self.fc(x))


class Deep1DCNN(nn.Module):
    def __init__(self, input_dim: int = 27) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_uniform_(module.weight, a=0.01, nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.01)
        x = self.pool(x).squeeze(2)
        x = self.dropout(x)
        return torch.sigmoid(self.fc(x))

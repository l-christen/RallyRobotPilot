import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):
    """
    Modèle CNN + LSTM pour séquence de 15 images.
    Pas d'état caché externe → le LSTM traite toute la séquence à chaque forward.
    Compatible avec le moteur d'inférence basé sur buffer (seq_len=15).
    """

    def __init__(self, img_height=224, img_width=160, lstm_hidden=256, lstm_layers=2):
        super().__init__()

        # --- CNN feature extractor ---
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.feature_dim = 256

        # --- LSTM ---
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.3 if lstm_layers > 1 else 0,
        )

        # --- Prediction heads ---
        self.raycast_head = nn.Sequential(
            nn.Linear(lstm_hidden, 128), nn.ReLU(), nn.Linear(128, 15)
        )

        self.speed_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        self.class_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64), nn.ReLU(), nn.Linear(64, 4)
        )

    # --------------------------------------------------------------------------------------------------

    def forward(self, x):
        """
        x : (B, 15, 3, 224, 160)
        """
        B, T, C, H, W = x.shape

        cnn_feats = []

        for i in range(T):
            f = self.cnn(x[:, i])              # (B,256,1,1)
            f = f.view(B, -1)                  # (B,256)
            cnn_feats.append(f)

        seq = torch.stack(cnn_feats, dim=1)    # (B,T,256)

        lstm_out, _ = self.lstm(seq)           # (B,T,H)
        last = lstm_out[:, -1]                 # (B,H)

        ray = self.raycast_head(last)          # (B,15)
        spd = self.speed_head(last)            # (B,1)
        logits = self.class_head(last)         # (B,4)

        return ray, spd, logits

    # --------------------------------------------------------------------------------------------------

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

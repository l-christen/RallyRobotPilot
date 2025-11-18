import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNetLiteLSTM(nn.Module):
    """
    Lightweight ResNet18 + LSTM model for driving.
    Now the action head uses:
        - LSTM hidden state
        - Predicted speed
        - Predicted raycasts
    This gives the controller knowledge of:
        - current visual context
        - current vehicle speed
        - spatial layout around the car (raycasts)
    """

    def __init__(self, lstm_hidden=64, lstm_layers=1):
        super().__init__()

        # --------------------
        # CNN feature extractor
        # --------------------
        base = resnet18(weights=None)

        # Keep only early layers (fast, lower resolution)
        self.cnn = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 128

        # Feature projection to 128-dim embedding
        self.embed = nn.Linear(self.feature_dim, 128)
        self.embed_dim = 128

        # --------------------
        # LSTM temporal encoder
        # --------------------
        self.lstm = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.0,
        )

        self.lstm_dropout = nn.Dropout(0.3)
        self.lstm_hidden = lstm_hidden

        # --------------------
        # Prediction heads
        # --------------------

        # Raycast regression head (15 distances)
        self.raycast_head = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 15)
        )

        # Speed regression head (1 scalar)
        self.speed_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # --------------------
        # NEW: Action head uses:
        #   - LSTM hidden
        #   - predicted speed
        #   - predicted raycasts
        # Dimension = lstm_hidden + 1 + 15
        # --------------------
        self.class_head = nn.Sequential(
            nn.Linear(lstm_hidden + 1 + 15, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    # ------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------
    def forward(self, x):
        """
        x : (B, T, C, H, W)
        Returns:
            (pred_raycasts, pred_speed, pred_actions)
        Same output format as before → train and infer unchanged.
        """
        B, T, C, H, W = x.shape
        feats = []

        # ----- Extract per-frame CNN features -----
        for t in range(T):
            f = self.cnn(x[:, t])          # (B,128,H',W')
            f = self.pool(f).view(B, -1)   # (B,128)
            f = self.embed(f)              # (B,128)
            feats.append(f)

        seq = torch.stack(feats, dim=1)    # (B,T,128)

        # ----- LSTM temporal encoding -----
        out, _ = self.lstm(seq)
        last = out[:, -1]                  # last timestep (B,lstm_hidden)
        last = self.lstm_dropout(last)

        # ----- Predict auxiliary tasks -----
        pred_raycasts = self.raycast_head(last)     # (B,15)
        pred_speed    = self.speed_head(last)       # (B,1)

        # ----- NEW: Concatenate auxiliary predictions -----
        augmented = torch.cat([last, pred_speed, pred_raycasts], dim=-1)
        pred_actions = self.class_head(augmented)   # (B,4)

        return pred_raycasts, pred_speed, pred_actions

    # ------------------------------------------------------
    # Utilities kept identical for infer.py compatibility
    # ------------------------------------------------------
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward_cnn(self, frame):
        """
        Used by inference engine: CNN only.
        """
        f = self.cnn(frame)
        f = self.pool(f).view(frame.size(0), -1)
        return self.embed(f)

    def forward_lstm(self, seq):
        """
        Used by inference engine: LSTM only.
        seq: (B,T,128)
        Returns last hidden state (B,lstm_hidden)
        """
        out, _ = self.lstm(seq)
        return out[:, -1]

    def forward_heads(self, last):
        """
        Used by inference engine: heads only.
        IMPORTANT: must reproduce the same logic as forward()
        but using already-provided last=LSTM_hidden.
        """
        pred_raycasts = self.raycast_head(last)
        pred_speed    = self.speed_head(last)

        augmented = torch.cat([last, pred_speed, pred_raycasts], dim=-1)
        pred_actions = self.class_head(augmented)

        return pred_raycasts, pred_speed, pred_actions

import torch
import torch.nn as nn
from torchvision.models import resnet18


class StackedResNetDriving(nn.Module):
    """
    CNN + TCN driving model avec forte régularisation :
        - Dropout2d dans CNN
        - LayerNorm + Dropout dans TCN
        - Dropout avant heads
        - Dropout dans heads
        - GELU activations
    """

    def __init__(self, num_frames=4, dropout=0.25):
        super().__init__()

        self.num_frames = num_frames
        self.dropout = dropout

        # --------------------
        # ResNet backbone
        # --------------------
        base = resnet18(weights=None)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = base.bn1
        self.relu    = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        # Dropout spatial (régularisation très forte)
        self.drop_cnn1 = nn.Dropout2d(p=dropout)
        self.drop_cnn2 = nn.Dropout2d(p=dropout)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 512

        self.embed_dim = 256
        self.embed = nn.Linear(self.feature_dim, self.embed_dim)

        # --------------------
        # TCN régularisé
        # --------------------
        self.tcn = nn.Sequential(
            nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Dropout global avant heads
        self.pre_head_dropout = nn.Dropout(dropout)

        # --------------------
        # Heads avec régularisation
        # --------------------
        self.raycast_head = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 15),
        )

        self.speed_head = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.class_head = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4),
        )


    # ============================================================
    # FORWARD
    # ============================================================
    def forward(self, x):
        B, T, C, H, W = x.shape

        if T > self.num_frames:
            x = x[:, -self.num_frames:]
            T = self.num_frames
        if T < self.num_frames:
            pad = self.num_frames - T
            x = torch.cat([x, x[:, -1:].repeat(1, pad, 1, 1, 1)], dim=1)

        feats = []
        for t in range(self.num_frames):
            feats.append(self.forward_cnn(x[:, t]))

        seq = torch.stack(feats, dim=1)       # (B,T,embed_dim)
        seq = seq.transpose(1, 2)            # (B,embed_dim,T)

        tc_out = self.tcn(seq)               # (B,embed_dim,T)

        fused = tc_out[:, :, -1]             # (B,embed_dim)
        fused = self.pre_head_dropout(fused)

        pred_raycasts = self.raycast_head(fused)
        pred_speed    = self.speed_head(fused)
        pred_actions  = self.class_head(fused)

        return pred_raycasts, pred_speed, pred_actions


    # ============================================================
    # CNN forward
    # ============================================================
    def forward_cnn(self, frame):
        x = self.conv1(frame)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.drop_cnn1(x)

        x = self.layer3(x)
        x = self.drop_cnn2(x)
        x = self.layer4(x)

        x = self.pool(x).view(frame.size(0), -1)
        feats = self.embed(x)

        return feats


    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

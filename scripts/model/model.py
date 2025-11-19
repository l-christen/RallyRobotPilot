import torch
import torch.nn as nn
from torchvision.models import resnet18


class StackedResNetDriving(nn.Module):
    """
    CNN + TCN driving model.
    - Entrée : séquence de T frames, shape (B, T, C, H, W)
    - CNN appliqué sur chaque frame
    - TCN (Temporal Conv Net) sur les embeddings de frames
    - Heads :
        * raycasts : 15 valeurs
        * speed    : 1 scalaire
        * actions  : 4 logits (forward/back/left/right)
    """

    def __init__(self, num_frames=4):
        super().__init__()

        self.num_frames = num_frames

        # --------------------
        # Backbone: ResNet18
        # --------------------
        base = resnet18(weights=None)

        in_channels = 3  # maintenant CNN appliqué PAR FRAME (pas concat channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # reprendre reste ResNet
        self.bn1      = base.bn1
        self.relu     = base.relu
        self.maxpool  = base.maxpool
        self.layer1   = base.layer1
        self.layer2   = base.layer2
        self.layer3   = base.layer3
        self.layer4   = base.layer4

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 512

        # projection embedding
        self.embed_dim = 256
        self.embed = nn.Linear(self.feature_dim, self.embed_dim)

        # --------------------
        # TCN temporel
        # --------------------
        self.tcn = nn.Sequential(
            nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
        )

        # --------------------
        # Heads
        # --------------------
        self.raycast_head = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 15),
        )

        self.speed_head = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.class_head = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )


    # ============================================================
    # FORWARD PRINCIPAL
    # ============================================================
    def forward(self, x):
        """
        x : (B, T, C, H, W)
        Retourne :
            pred_raycasts : (B, 15)
            pred_speed    : (B, 1)
            pred_actions  : (B, 4)
        """
        B, T, C, H, W = x.shape

        # ajuster longueur T
        if T > self.num_frames:
            x = x[:, -self.num_frames:]
            T = self.num_frames

        if T < self.num_frames:
            pad_frames = self.num_frames - T
            x = torch.cat([x, x[:, -1:].repeat(1, pad_frames, 1, 1, 1)], dim=1)
            T = self.num_frames

        # ---------------------------------------------------------
        # CNN appliqué *frame par frame*
        # ---------------------------------------------------------
        feats_list = []
        for t in range(T):
            ft = self.forward_cnn(x[:, t])   # (B, embed_dim)
            feats_list.append(ft)

        # (B,T,embed_dim)
        seq = torch.stack(feats_list, dim=1)

        # TCN attend (B, embed_dim, T)
        seq_tcn = seq.transpose(1, 2)

        # passé dans TCN → toujours (B, embed_dim, T)
        tc_out = self.tcn(seq_tcn)

        # on prend la dernière position temporelle (plus stable)
        fused = tc_out[:, :, -1]   # (B, embed_dim)

        # Heads
        pred_raycasts = self.raycast_head(fused)
        pred_speed    = self.speed_head(fused)
        pred_actions  = self.class_head(fused)

        return pred_raycasts, pred_speed, pred_actions


    # ============================================================
    # UTILITAIRES
    # ============================================================
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    def forward_cnn(self, frame):
        """
        frame : (B,3,H,W)
        Retourne embedding (B, embed_dim)
        """
        x = self.conv1(frame)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x).view(x.size(0), -1)
        feats = self.embed(x)
        return feats


    def forward_heads(self, feats):
        pred_raycasts = self.raycast_head(feats)
        pred_speed    = self.speed_head(feats)
        pred_actions  = self.class_head(feats)
        return pred_raycasts, pred_speed, pred_actions



# ============================================================
# DUMMY CHECKPOINT MAKER
# ============================================================
if __name__ == "__main__":

    out_path = "checkpoints/dummy.pth"
    model = StackedResNetDriving(num_frames=4)

    checkpoint = {
        "epoch": 0,
        "val_loss": 999.0,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
    }

    torch.save(checkpoint, out_path)
    print(f"✓ Dummy checkpoint saved to {out_path}")
    print(f"#params = {model.get_num_parameters():,}")

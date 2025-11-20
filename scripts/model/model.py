import torch
import torch.nn as nn


class StackedResNetDriving(nn.Module):
    """
    CNN + TCN driving model (version allégée pour ~94k exemples).
    - Entrée : séquence de T frames, shape (B, T, C, H, W)
    - CNN léger appliqué sur chaque frame
    - TCN (Temporal Conv Net) sur les embeddings de frames
    - Heads :
        * raycasts : 15 valeurs
        * speed    : 1 scalaire
        * actions  : 4 logits (forward/back/left/right)
    
    Paramètres : ~250-300k (vs ~11M pour ResNet18 complet)
    """

    def __init__(self, num_frames=4):
        super().__init__()

        self.num_frames = num_frames

        # --------------------
        # Backbone: CNN léger (inspiré MobileNet)
        # --------------------
        # Au lieu de ResNet18, on utilise une architecture plus compacte
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Blocs convolutifs progressifs
        self.layer1 = self._make_layer(32, 64, stride=2)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 256

        # projection embedding (réduite)
        self.embed_dim = 128  # réduit de 256 à 128
        self.embed = nn.Linear(self.feature_dim, self.embed_dim)

        # --------------------
        # TCN temporel (simplifié)
        # --------------------
        self.tcn = nn.Sequential(
            nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU()
        )

        # --------------------
        # Heads (simplifiés)
        # --------------------
        self.raycast_head = nn.Sequential(
            nn.Linear(self.embed_dim, 64),  # réduit de 128 à 64
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 15),
        )

        self.speed_head = nn.Sequential(
            nn.Linear(self.embed_dim, 32),  # réduit de 64 à 32
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.class_head = nn.Sequential(
            nn.Linear(self.embed_dim, 64),  # réduit de 128 à 64
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 4),
        )

    def _make_layer(self, in_channels, out_channels, stride=1):
        """Bloc convolutif simple avec BatchNorm et résiduel optionnel"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                     stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
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
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
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
    
    # Test forward pass
    dummy_input = torch.randn(2, 4, 3, 224, 224)
    raycasts, speed, actions = model(dummy_input)
    print(f"✓ Forward pass OK")
    print(f"  - Raycasts: {raycasts.shape}")
    print(f"  - Speed: {speed.shape}")
    print(f"  - Actions: {actions.shape}")
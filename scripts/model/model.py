import torch
import torch.nn as nn
from torchvision.models import resnet18


class StackedResNetDriving(nn.Module):
    """
    CNN-only driving model.
    - Entrée : séquence de T frames, shape (B, T, C, H, W)
    - On concatène les T frames sur le canal : (B, 3*T, H, W)
    - Backbone : ResNet18 complet (relativement puissant)
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

        # Adapter la première couche pour 3 * T channels
        in_channels = 3 * num_frames
        self.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # On reprend le reste du ResNet
        self.bn1   = base.bn1
        self.relu  = base.relu
        self.maxpool = base.maxpool
        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3
        self.layer4  = base.layer4

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 512   # ResNet18 output dim

        # Optionnel : projeter les features dans un espace plus compact
        self.embed_dim = 256
        self.embed = nn.Linear(self.feature_dim, self.embed_dim)

        # --------------------
        # Heads de prédiction
        # --------------------
        # Raycasts (15)
        self.raycast_head = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 15),
        )

        # Speed (1)
        self.speed_head = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
        # Actions (4)
        self.class_head = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    # --------------------
    # Forward principal
    # --------------------
    def forward(self, x):
        """
        x : (B, T, C, H, W)
        Retourne :
            pred_raycasts : (B, 15)
            pred_speed    : (B, 1)
            pred_actions  : (B, 4)
        """
        B, T, C, H, W = x.shape

        # Si T > num_frames, on garde juste les dernières frames
        if T > self.num_frames:
            x = x[:, -self.num_frames:]
            T = self.num_frames

        # Si T < num_frames, tu peux :
        # - soit pad avec la première/dernière frame
        # - soit assert. Ici je duplique la dernière frame.
        if T < self.num_frames:
            pad_frames = self.num_frames - T
            last = x[:, -1:].repeat(1, pad_frames, 1, 1, 1)
            x = torch.cat([x, last], dim=1)
            T = self.num_frames

        # Concaténation sur le canal : (B, T*C, H, W)
        x = x.view(B, T * C, H, W)

        # Backbone ResNet18 complet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global pooling + embedding
        x = self.pool(x).view(B, -1)   # (B, 512)
        feats = self.embed(x)          # (B, embed_dim)

        # Heads auxiliaires
        pred_raycasts = self.raycast_head(feats)   # (B, 15)
        pred_speed    = self.speed_head(feats)     # (B, 1)

        pred_actions = self.class_head(feats) # (B, 4)

        return pred_raycasts, pred_speed, pred_actions

    # --------------------
    # Utilitaires
    # --------------------
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward_cnn(self, frame_stack):
        """
        Optionnel : utilisé en inference si tu veux passer
        directement un stack de frames déjà concaténé :
            frame_stack : (B, 3*num_frames, H, W)
        Retourne : embedding (B, embed_dim)
        """
        x = frame_stack

        x = self.conv1(x)
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
        """
        Si tu as déjà l'embedding `feats` (B, embed_dim) :
        retourne (pred_raycasts, pred_speed, pred_actions)
        """
        pred_raycasts = self.raycast_head(feats)
        pred_speed    = self.speed_head(feats)
        augmented = torch.cat([feats, pred_raycasts, pred_speed], dim=-1)
        pred_actions = self.class_head(augmented)
        return pred_raycasts, pred_speed, pred_actions

if __name__ == "__main__":

    # Output filename
    out_path = "checkpoints/dummy.pth"

    # Create model (default num_frames=4)
    model = StackedResNetDriving(num_frames=4)

    # fake optimizer just for format consistency
    optimizer_state = {}

    checkpoint = {
        "epoch": 0,
        "val_loss": 999.0,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer_state,
    }

    torch.save(checkpoint, out_path)
    print(f"✓ Dummy checkpoint saved to {out_path}")
    print(f"  #params = {model.get_num_parameters():,}")
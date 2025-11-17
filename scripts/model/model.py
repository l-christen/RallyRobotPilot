import torch
import torch.nn as nn
from torchvision.models import resnet18

# TODO : Add 2 separated forward methods: one for CNN feature extraction only, one for LSTM + heads.

class ResNetLiteLSTM(nn.Module):
    """
    CNN (ResNet18 tronqué) + LSTM
    Optimisé pour séquences longues (40 frames) et images 3×224×160.
    """

    def __init__(self, lstm_hidden=256, lstm_layers=1):
        super().__init__()

        # ----------- CNN FEATURE EXTRACTOR (ResNet18 sans les blocs lourds) -----------
        base = resnet18(weights=None)

        # On garde juste les 3 premiers blocs (rapide + pas trop lourd)
        self.cnn = nn.Sequential(
            base.conv1,     # (B,64,H/2,W/2)
            base.bn1,
            base.relu,
            base.maxpool,   # (B,64,H/4,W/4)
            base.layer1,    # (B,64,H/4,W/4)
            base.layer2     # (B,128,H/8,W/8)
        )

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.feature_dim = 128

        # Projection en embedding 128
        self.embed = nn.Linear(self.feature_dim, 128)
        self.embed_dim = 128

        # ----------- LSTM TEMPORAL MODELING -----------
        self.lstm = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.2 if lstm_layers > 1 else 0,
        )

        # ----------- HEADS (identiques à ton modèle) -----------
        self.raycast_head = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 15)
        )

        self.speed_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.class_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )


    def forward(self, x):
        """
        x : (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape
        feats = []

        for t in range(T):
            f = self.cnn(x[:, t])        # (B,128,H',W')
            f = self.pool(f).view(B, -1) # (B,128)
            f = self.embed(f)            # (B,128)
            feats.append(f)

        seq = torch.stack(feats, dim=1)  # (B,T,128)
        out, _ = self.lstm(seq)
        last = out[:, -1]

        return (
            self.raycast_head(last),
            self.speed_head(last),
            self.class_head(last)
        )

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
if __name__ == "__main__":
    model = ResNetLiteLSTM()

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": 0,
        "val_loss": 0.0,
    }

    torch.save(checkpoint, "checkpoints/dummy.pth")
    print("dummy.pth créé avec succès !")

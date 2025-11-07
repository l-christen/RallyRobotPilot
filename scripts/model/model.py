import torch
import torch.nn as nn

class CNNLSTMModel(nn.Module):
    """
    Modèle CNN + LSTM pour prédire:
    - 15 raycasts
    - 1 vitesse
    - 4 classifications (w, a, s, d)
    
    Entrée: séquence de 15 images
    """
    def __init__(self, img_height=160, img_width=224, lstm_hidden=256, lstm_layers=2):
        super(CNNLSTMModel, self).__init__()
        
        # CNN Feature Extractor (inspired by small ResNet)
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.cnn_output_dim = 256
        
        # LSTM pour capturer la temporalité
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.3 if lstm_layers > 1 else 0
        )
        
        # Têtes de prédiction
        # 1. Raycasts (15 valeurs de régression)
        self.raycast_head = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 15)
        )
        
        # 2. Vitesse (1 valeur de régression)
        self.speed_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
        # 3. Classification multi-labels (w, a, s, d)
        self.classification_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 4)
        )
        
    def forward(self, x):
        """
        Args:
            x: Tensor de forme (batch_size, seq_len=15, channels=3, height, width)
        
        Returns:
            raycasts: (batch_size, 15)
            speed: (batch_size, 1)
            classification: (batch_size, 4)
        """
        batch_size, seq_len, c, h, w = x.size()
        
        # Extraire les features CNN pour chaque frame
        cnn_features = []
        for t in range(seq_len):
            frame = x[:, t, :, :, :]  # (batch_size, 3, h, w)
            features = self.cnn(frame)  # (batch_size, 256, 1, 1)
            features = features.view(batch_size, -1)  # (batch_size, 256)
            cnn_features.append(features)
        
        # Stack les features temporelles
        cnn_features = torch.stack(cnn_features, dim=1)  # (batch_size, seq_len, 256)
        
        # Passer par le LSTM
        lstm_out, (h_n, c_n) = self.lstm(cnn_features)
        
        # Utiliser la dernière sortie du LSTM
        last_output = lstm_out[:, -1, :]  # (batch_size, lstm_hidden)
        
        # Prédictions pour chaque tête
        raycasts = self.raycast_head(last_output)  # (batch_size, 15)
        speed = self.speed_head(last_output)  # (batch_size, 1)
        classification_logits = self.classification_head(last_output)  # (batch_size, 4)
        
        return raycasts, speed, classification_logits
    
    def get_num_parameters(self):
        """Retourne le nombre de paramètres du modèle"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test du modèle
    model = CNNLSTMModel(img_height=224, img_width=224)
    print(f"Nombre de paramètres: {model.get_num_parameters():,}")
    
    # Test avec un batch
    batch_size = 4
    seq_len = 15
    dummy_input = torch.randn(batch_size, seq_len, 3, 224, 224)
    
    raycasts, speed, classification = model(dummy_input)
    
    print(f"Raycasts shape: {raycasts.shape}")  # (4, 15)
    print(f"Speed shape: {speed.shape}")  # (4, 1)
    print(f"Classification shape: {classification.shape}")  # (4, 4)
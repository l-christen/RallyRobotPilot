import torch
import numpy as np
from model.model import ResNetLiteLSTM

def scale_image(img):
    """
    img : torch.Tensor ou np.ndarray, shape (C,H,W)
    - Si img est float32 -> on suppose valeurs 0–255, on divise par 255.
    - Si img est uint8   -> convertit en float32 puis /255
    Retour : float32 dans [0,1]
    """
    
    # Si numpy
    if isinstance(img, np.ndarray):
        if img.dtype == np.float32:
            return img / 255.0
        elif img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        else:
            raise ValueError(f"Unsupported dtype: {img.dtype}")

    # Si tensor PyTorch
    if isinstance(img, torch.Tensor):
        if img.dtype == torch.float32:
            return img / 255.0
        elif img.dtype == torch.uint8:
            return img.float() / 255.0
        else:
            raise ValueError(f"Unsupported tensor dtype: {img.dtype}")

    raise TypeError("img must be NumPy array or torch.Tensor")


def load_model(checkpoint_path, device='cuda'):
    """
    Charge un modèle CNN-LSTM depuis un checkpoint.
    
    Args:
        checkpoint_path: Chemin vers le fichier .pth
        img_height: Hauteur des images d'entrée
        img_width: Largeur des images d'entrée
        device: 'cuda' ou 'cpu'
    
    Returns:
        model: Modèle chargé en mode eval
    """
    model = ResNetLiteLSTM()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    val_loss = checkpoint.get('val_loss', 'unknown')
    print(f"✓ Modèle chargé depuis {checkpoint_path}")
    print(f"  Epoch: {epoch}, Val Loss: {val_loss}")
    
    return model


class SequenceInferenceEngine:
    """
    Moteur d'inférence pour le modèle CNN-LSTM.
    Gère le buffering de séquences et l'inférence avec gestion du warm-up.
    """
    
    def __init__(self, model, seq_len=10, device='cuda'):
        """
        Args:
            model: Modèle CNNLSTMModel déjà chargé
            seq_len: Longueur de séquence attendue (doit correspondre à l'entraînement)
            device: Device PyTorch
        """
        self.model = model
        self.seq_len = seq_len
        self.device = device
        self.frame_buffer = []
        
        # Statistiques
        self.total_frames = 0
        self.total_inferences = 0
        self.is_ready = False
        
        print(f"✓ Moteur d'inférence initialisé (seq_len={seq_len}, device={device})")
    
    def add_frame(self, image):
        """
        Ajoute une frame au buffer. Au lieu de stocker l'image,
        on calcule directement son embedding CNN et on le stocke.

        Args:
            image: numpy (H,W,3) uint8

        Returns:
            bool: True si la séquence est prête pour prédiction
        """

        # ----------- Prétraitement minimal -----------
        image = np.transpose(image, (2, 0, 1))         # (C,H,W)
        img_tensor = torch.from_numpy(scale_image(image)).float().to(self.device)
        img_tensor = img_tensor.unsqueeze(0)           # (1,C,H,W)

        # ----------- Embedding CNN unique -----------
        with torch.no_grad():
            embed = self.model.forward_cnn(img_tensor) # (1,128)

        # On stocke l'embedding uniquement (pas les images)
        self.frame_buffer.append(embed.squeeze(0))     # (128,)
        self.total_frames += 1

        # ----------- Fenêtre glissante -----------
        if len(self.frame_buffer) > self.seq_len:
            self.frame_buffer.pop(0)

        # ----------- Check readiness -----------
        self.is_ready = len(self.frame_buffer) == self.seq_len
        return self.is_ready



    @torch.no_grad()
    def predict(self, threshold=1):
        """
        Prédiction basée sur les embeddings déjà en cache.
        """

        if not self.is_ready:
            return None

        # ----------- Stack embeddings : (T,128) → (1,T,128) -----------
        seq = torch.stack(self.frame_buffer).unsqueeze(0).to(self.device)

        # ----------- LSTM puis heads (CNN déjà fait dans add_frame) -----------
        last = self.model.forward_lstm(seq)          # (1,hidden)
        pred_ray, pred_speed, pred_class = self.model.forward_heads(last)
        
        # ----------- Post-traitement -----------
        probabilities = torch.sigmoid(pred_class).cpu().numpy().flatten()
        print(probabilities)
        controls = (probabilities > threshold).tolist()

        raycasts = pred_ray.cpu().numpy().flatten()
        speed = pred_speed.cpu().item()

        self.total_inferences += 1

        return {
            "controls": [bool(c) for c in controls],
            "probabilities": probabilities.tolist(),
            "raycasts": raycasts,
            "speed": speed,
            "ready": True,
        }
    
    def predict_controls_only(self, threshold=0.95):
        """
        Version simplifiée qui ne retourne que les contrôles.
        
        Returns:
            tuple: (forward, back, left, right) ou None si pas prêt
        """
        result = self.predict(threshold)
        if result is None:
            return None
        return tuple(result['controls'])
    
    def reset(self):
        """Vide le buffer (utile en cas de reset de partie)."""
        self.frame_buffer = []
        self.is_ready = False
        print("[!] Buffer réinitialisé")
    
    def get_status(self):
        """Retourne l'état actuel du moteur."""
        return {
            'buffer_size': len(self.frame_buffer),
            'required_size': self.seq_len,
            'ready': self.is_ready,
            'total_frames': self.total_frames,
            'total_inferences': self.total_inferences,
            'progress': len(self.frame_buffer) / self.seq_len
        }


def create_inference_engine(checkpoint_path, seq_len=10, device='cuda'):
    """
    Fonction helper pour créer un moteur d'inférence complet.
    
    Args:
        checkpoint_path: Chemin vers le modèle
        seq_len: Longueur de séquence
        device: Device PyTorch
    
    Returns:
        SequenceInferenceEngine: Moteur prêt à l'emploi
    """
    model = load_model(checkpoint_path, device=device)
    engine = SequenceInferenceEngine(model, seq_len=seq_len, device=device)
    return engine
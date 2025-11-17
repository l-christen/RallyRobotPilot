import torch
import numpy as np
from model.model import ResNetLiteLSTM

# TODO : Add a cache for the embeding of frames already seen.

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
    
    def __init__(self, model, seq_len=40, device='cuda'):
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
        Ajoute une frame au buffer.
        
        Args:
            image: Image numpy (H, W, 3) en uint8 [0-255]
        
        Returns:
            bool: True si le buffer est prêt pour l'inférence
        """
        # Prétraiter l'image (transpose + scale)
        image = np.transpose(image, (2, 0, 1))  # (C,H,W)
        img_tensor = torch.from_numpy(scale_image(image)).float()
        
        # Ajouter au buffer
        self.frame_buffer.append(img_tensor)
        self.total_frames += 1
        
        # Maintenir la taille du buffer
        if len(self.frame_buffer) > self.seq_len:
            self.frame_buffer.pop(0)
        
        # Vérifier si on est prêt
        self.is_ready = len(self.frame_buffer) == self.seq_len
        
        return self.is_ready
    
    @torch.no_grad()
    def predict(self, threshold=0.5):
        """
        Effectue une prédiction sur la séquence actuelle.
        
        Args:
            threshold: Seuil de décision pour la classification binaire
        
        Returns:
            dict: {
                'controls': [forward, back, left, right] (bool),
                'probabilities': [p_fwd, p_back, p_left, p_right] (float),
                'raycasts': array de 15 distances,
                'speed': vitesse prédite,
                'ready': bool indiquant si la prédiction est fiable
            }
            ou None si pas assez de frames
        """
        if not self.is_ready:
            return None
        
        # Stack en séquence: (seq_len, 3, H, W) → (1, seq_len, 3, H, W)
        sequence = torch.stack(self.frame_buffer).unsqueeze(0).to(self.device)
        
        # Forward pass
        pred_raycasts, pred_speed, pred_class = self.model(sequence)
        
        # Post-traitement
        probabilities = torch.sigmoid(pred_class).cpu().numpy().flatten()
        controls = (probabilities > threshold).tolist()
        
        raycasts = pred_raycasts.cpu().numpy().flatten()
        speed = pred_speed.cpu().item()
        
        self.total_inferences += 1
        
        return {
            'controls': [bool(c) for c in controls],  # [fwd, back, left, right]
            'probabilities': probabilities.tolist(),
            'raycasts': raycasts,
            'speed': speed,
            'ready': True
        }
    
    def predict_controls_only(self, threshold=0.5):
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


def create_inference_engine(checkpoint_path, seq_len=40, device='cuda'):
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
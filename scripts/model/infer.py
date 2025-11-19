import torch
import numpy as np
from model.model import StackedResNetDriving


# ============================
# Image utils
# ============================
def scale_image(img_chw):
    """
    img_chw : numpy (C,H,W) float32 or uint8
    Returns float32 in [0,1]
    """
    if isinstance(img_chw, np.ndarray):
        if img_chw.dtype == np.uint8:
            return img_chw.astype(np.float32) / 255.0
        elif img_chw.dtype == np.float32:
            return img_chw / 255.0
    elif isinstance(img_chw, torch.Tensor):
        if img_chw.dtype == torch.uint8:
            return img_chw.float() / 255.0
        elif img_chw.dtype == torch.float32:
            return img_chw / 255.0

    raise TypeError("scale_image expects float32 or uint8 ndarray/tensor.")


# ============================
# Load model
# ============================
def load_model(checkpoint_path, device="cuda"):
    model = StackedResNetDriving(num_frames=2)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"✓ Loaded {checkpoint_path}")
    print(f"  epoch={ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss','?')}")
    return model


# ============================
# Inference Engine
# ============================
class SequenceInferenceEngine:
    def __init__(self, model, seq_len=2, device="cuda"):
        self.model = model
        self.seq_len = seq_len
        self.device = device

        self.buffer = []     # stores (C,H,W) float32 scaled
        self.ready = False
        self.total_frames = 0
        self.total_infer = 0

        print(f"✓ Inference engine ready (seq_len={seq_len})")

    # ------------------------------------------
    # Add frame (H,W,3) float32 or uint8
    # ------------------------------------------
    def add_frame(self, img_hwc):
        """
        img_hwc : numpy array (H,W,3), float32 (0-255) or uint8
        """

        # Convert -> CHW float32
        chw = np.transpose(img_hwc, (2, 0, 1))
        chw = scale_image(chw)   # float32 [0,1]

        tensor = torch.from_numpy(chw).float()  # (C,H,W)
        self.buffer.append(tensor)
        self.total_frames += 1

        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)

        self.ready = (len(self.buffer) == self.seq_len)
        return self.ready

    # ------------------------------------------
    # Predict
    # ------------------------------------------
    @torch.no_grad()
    def predict(self):
        if not self.ready:
            return None

        # Build (1,T,C,H,W)
        seq = torch.stack(self.buffer).unsqueeze(0).to(self.device)

        pred_ray, pred_speed, pred_actions = self.model(seq)

        probs = torch.sigmoid(pred_actions)[0].cpu().numpy()
        print(probs)
        ray = pred_ray[0].cpu().numpy()
        speed = pred_speed[0].item()

        # default thresholds
        thr = [0.4, 0.5, 0.6, 0.6]
        controls = [bool(probs[i] > thr[i]) for i in range(4)]

        self.total_infer += 1

        return {
            "probabilities": probs.tolist(),
            "controls": controls,
            "raycasts": ray,
            "speed": speed,
            "ready": True,
        }

    # ------------------------------------------
    def predict_controls_only(self):
        out = self.predict()
        if out is None:
            return None
        return tuple(out["controls"])

    # ------------------------------------------
    def reset(self):
        self.buffer = []
        self.ready = False
        print("Buffer reset.")

    # ------------------------------------------
    def get_status(self):
        return {
            "buffer": len(self.buffer),
            "required": self.seq_len,
            "ready": self.ready,
            "total_frames": self.total_frames,
            "total_inferences": self.total_infer,
        }


def create_inference_engine(checkpoint_path, seq_len=2, device="cuda"):
    model = load_model(checkpoint_path, device=device)
    return SequenceInferenceEngine(model, seq_len=seq_len, device=device)

# autopilot_engine.py

from model.infer import create_inference_engine
from .sensing_message import SensingSnapshot
import time

class AutopilotEngine:
    def __init__(self, checkpoint_path, device="auto", seq_len=20):
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.engine = create_inference_engine(
            checkpoint_path=checkpoint_path,
            device=device,
            seq_len=seq_len,
        )

        self.device = device
        self.last_inference_time = 0
        print(f"[✓] Autopilot engine ready on {device}")

    def process_snapshot(self, snapshot):
        """Return (forward, back, left, right) or None if not ready."""
        is_ready = self.engine.add_frame(snapshot.image)

        if not is_ready:
            return None

        result = self.engine.predict(threshold=0.5)
        if result is None:
            return None

        return tuple(result["controls"])

from model.infer import create_inference_engine
from .sensing_message import SensingSnapshot


class AutopilotEngine:
    def __init__(self, checkpoint_path, device="auto", seq_len=4):
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.engine = create_inference_engine(
            checkpoint_path=checkpoint_path,
            seq_len=seq_len,
            device=device,
        )

        self.device = device
        print(f"[✓] Autopilot engine initialized on {device}")

    def process_snapshot(self, snapshot):
        """
        snapshot.image : numpy (H,W,3) float32 coming from Panda3D
        Returns : tuple(bool,bool,bool,bool) or None
        """

        ready = self.engine.add_frame(snapshot.image)

        if not ready:
            return None

        result = self.engine.predict()
        if result is None:
            return None

        return tuple(result["controls"])

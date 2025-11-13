from PyQt6 import QtWidgets
import sys
import torch
from data_collector import DataCollectionUI
from model.infer import infer_single_snapshot, load_model
import time

class AutopilotProcessor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = load_model('checkpoints/best_model.pth', device=self.device)
        self.hidden = None
        self.prev_states = {"forward": False, "back": False, "left": False, "right": False}
        print(f"[✓] Autopilot initialisé sur {self.device}")
        self.last_infer_time = 0.0

    def process_message(self, message, data_collector):
        if time.time() - self.last_infer_time < 0.13:  # toutes les 100 ms
            return
        self.last_infer_time = time.time()
        try:
            pred_controls, self.hidden = infer_single_snapshot(
                self.model, message, hidden=self.hidden, device=self.device
            )

            # compatibilité tensor / numpy
            if hasattr(pred_controls, "cpu"):
                pred_controls = pred_controls.cpu().numpy()
            pressed = (pred_controls > 0.5).tolist()

            commands = {
                "forward": bool(pressed[0]),
                "back": bool(pressed[1]),
                "left": bool(pressed[2]),
                "right": bool(pressed[3]),
            }

            # --- envoi uniquement les transitions valides
            for cmd, new_state in commands.items():
                old_state = self.prev_states.get(cmd, False)
                if new_state != old_state:
                    data_collector.onCarControlled(cmd, new_state)
                    self.prev_states[cmd] = new_state

        except Exception as e:
            print(f"[X] Erreur autopilot : {e}")


if __name__ == "__main__":
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)

    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)
    autopilot = AutopilotProcessor()
    window = DataCollectionUI(message_processing_callback=autopilot.process_message)
    window.show()
    sys.exit(app.exec())

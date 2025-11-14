# input_manager.py

from ursina import held_keys

class InputMode:
    HUMAN = "human"
    AUTOPILOT = "autopilot"

class InputManager:
    def __init__(self):
        self.mode = InputMode.HUMAN
        self.autopilot_output = {"forward": False, "back": False, "left": False, "right": False}

    # Toggle autopilot
    def toggle_autopilot(self):
        if self.mode == InputMode.HUMAN:
            self.mode = InputMode.AUTOPILOT
        else:
            self.mode = InputMode.HUMAN
        print(f"[INPUT] Mode → {self.mode}")

    # Called by autopilot engine
    def set_autopilot_output(self, fwd, back, left, right):
        self.autopilot_output = {
            "forward": bool(fwd),
            "back": bool(back),
            "left": bool(left),
            "right": bool(right)
        }

    def get_inputs(self):
        if self.mode == InputMode.HUMAN:
            forward  = bool(held_keys['w'] or held_keys["up arrow"])
            back     = bool(held_keys['s'] or held_keys["down arrow"])
            left     = bool(held_keys['a'] or held_keys["left arrow"])
            right    = bool(held_keys['d'] or held_keys["right arrow"])
            return forward, back, left, right
        
        # Autopilot
        ap = self.autopilot_output
        return ap["forward"], ap["back"], ap["left"], ap["right"]

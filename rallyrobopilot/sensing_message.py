"""
    SensingSnapshot is a packing/unpacking class for diverse car simulation related information
"""
class SensingSnapshot:
    def __init__(self):
        #   Forward - Backward - Left - Right
        self.current_controls = (0,0,0,0)
        self.car_position = (0,0,0)
        self.car_speed = 0
        self.car_angle = 0
        self.raycast_distances = [0]
        self.image = None
        self.timestamp = 0.0

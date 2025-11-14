import os
import time
import pickle
import lzma
import numpy as np

from ursina import *
from ursina import curve
from math import pow, atan2, cos, sin

from .particles import Particles, TrailRenderer
from .sensing_message import SensingSnapshot
from .input_manager import InputManager, InputMode
from .autopilot_engine import AutopilotEngine

sign = lambda x: -1 if x < 0 else (1 if x > 0 else 0)
Text.default_resolution = 1080 * Text.size


class Car(Entity):
    """
    Car d'origine avec :
      - Physique, collisions, caméra, trails, UI conservés
      - Mode humain / autopilot via InputManager
      - Enregistrement de toutes les frames (SensingSnapshot) quand activé
      - CORRIGÉ: steering et collision handling identiques au document 2
    """

    def __init__(
        self,
        position=(0, 0, 4),
        rotation=(0, 0, 0),
        topspeed=30,
        acceleration=0.35,
        braking_strength=30,
        friction=1.5,
        camera_speed=8,
        model_path="checkpoints/dummy.pth",
    ):
        super().__init__(
            model="assets/cars/sports-car.obj",
            texture="assets/cars/garage/sports-car/sports-red.png",
            collider="sphere",
            position=position,
            rotation=rotation,
        )

        # --- NEW: Autopilot / input routing ---
        self.input_manager = InputManager()
        self.autopilot = AutopilotEngine(model_path)

        # Controls
        self.controls = "wasd"

        # Car's values
        self.speed = 0
        self.velocity_y = 0
        self.rotation_speed = 0
        self.max_rotation_speed = 1.6
        self.steering_amount = 8
        self.topspeed = topspeed
        self.braking_strenth = braking_strength
        self.camera_speed = camera_speed
        self.acceleration = acceleration
        self.friction = friction
        self.turning_speed = 5
        self.pivot_rotation_distance = 1

        self.reset_position = (0, 0, 0)
        self.reset_orientation = (0, 0, 0)

        # Camera Follow
        self.camera_angle = "top"
        self.camera_offset = (0, 30, -35)
        self.camera_rotation = 40
        self.camera_follow = False
        self.change_camera = False
        self.c_pivot = Entity()
        self.camera_pivot = Entity(parent=self.c_pivot, position=self.camera_offset)

        # Pivot for drifting
        self.pivot = Entity()
        self.pivot.position = self.position
        self.pivot.rotation = self.rotation
        self.drifting = False

        # Car Type
        self.car_type = "sports"

        # Particles
        self.particle_pivot = Entity(parent=self)
        self.particle_pivot.position = (0, -1, -2)

        # TrailRenderer
        self.trail_pivot = Entity(parent=self, position=(0, -1, 2))
        self.trail_renderer1 = TrailRenderer(
            parent=self.particle_pivot,
            position=(0.8, -0.2, 0),
            color=color.black,
            alpha=0,
            thickness=7,
            length=200,
        )
        self.trail_renderer2 = TrailRenderer(
            parent=self.particle_pivot,
            position=(-0.8, -0.2, 0),
            color=color.black,
            alpha=0,
            thickness=7,
            length=200,
        )
        self.trail_renderer3 = TrailRenderer(
            parent=self.trail_pivot,
            position=(0.8, -0.2, 0),
            color=color.black,
            alpha=0,
            thickness=7,
            length=200,
        )
        self.trail_renderer4 = TrailRenderer(
            parent=self.trail_pivot,
            position=(-0.8, -0.2, 0),
            color=color.black,
            alpha=0,
            thickness=7,
            length=200,
        )
        self.trails = [
            self.trail_renderer1,
            self.trail_renderer2,
            self.trail_renderer3,
            self.trail_renderer4,
        ]
        self.start_trail = True

        # Collision
        self.copy_normals = False
        self.hitting_wall = False

        self.track = None

        # Graphics
        self.graphics = "fancy"

        # Stopwatch/Timer
        self.timer_running = False
        self.count = 0.0
        self.last_count = self.count
        self.reset_count = 0.0
        self.timer = Text(
            text="",
            origin=(0, 0),
            size=0.05,
            scale=(1, 1),
            position=(-0.7, 0.43),
        )
        self.laps_text = Text(
            text="",
            origin=(0, 0),
            size=0.05,
            scale=(1.1, 1.1),
            position=(0, 0.43),
        )
        self.reset_count_timer = Text(
            text=str(round(self.reset_count, 1)),
            origin=(0, 0),
            size=0.05,
            scale=(1, 1),
            position=(-0.7, 0.43),
        )
        self.timer.disable()
        self.laps_text.disable()
        self.reset_count_timer.disable()

        self.gamemode = "race"
        self.start_time = False
        self.laps = 0
        self.laps_hs = 0
        self.anti_cheat = 1

        # Bools
        self.driving = False
        self.braking = False

        # Multiplayer
        self.multiplayer = False
        self.multiplayer_update = False
        self.server_running = False

        # Shows whether you are connected to a server or not
        self.connected_text = True
        self.disconnected_text = True

        # Camera shake
        self.shake_amount = 0.1
        self.can_shake = False
        self.camera_shake_option = True

        self.username_text = "Username"

        self.model_path = str(self.model).replace("render/scene/car/", "")
        invoke(self.update_model_path, delay=1)

        self.multiray_sensor = None

        # === Recording (toutes les frames quand activé) ===
        self._record_enabled = False
        self._record_buffer = []          # list[SensingSnapshot]
        self._record_dir = "data"
        os.makedirs(self._record_dir, exist_ok=True)

        print("[CAR] Car initialized (with autopilot & recording)")

    # ===== Public API to toggle recording =====
    def enable_recording(self, enabled: bool = True, period_hz: float = None, out_dir: str = "data"):
        """
        Active/désactive l'enregistrement.
        NOTE: on ignore period_hz maintenant → on enregistre chaque frame.
        """
        self._record_enabled = enabled
        self._record_dir = out_dir or "data"
        os.makedirs(self._record_dir, exist_ok=True)
        print(f"[REC] Recording {'enabled' if enabled else 'disabled'} (all frames)")

    # ===== Track/Car setup (compatibles avec game_launcher) =====
    def set_track(self, track):
        self.track = track
        self.reset_position = track.car_default_reset_position
        self.reset_orientation = track.car_default_reset_orientation
        self.position = self.reset_position
        self.rotation_y = self.reset_orientation[1]

    def sports_car(self):
        self.car_type = "sports"
        self.model = "assets/cars/sports-car.obj"
        self.texture = "assets/cars/garage/sports-car/sports-red.png"
        self.topspeed = 50
        self.minspeed = -15
        self.acceleration = 25
        self.braking_strenth = 50
        self.turning_speed = 6
        self.max_rotation_speed = 1.6
        self.steering_amount = 9
        self.particle_pivot.position = (0, -1, -1.5)
        self.trail_pivot.position = (0, -1, 1.5)

    # ===== Camera =====
    def update_camera(self):
        if self.camera_follow:
            if self.change_camera:
                camera.rotation_x = 35
                self.camera_rotation = 40
            self.camera_offset = (0, 60, -70)
            self.camera_speed = 4
            self.change_camera = False
            camera.world_position = self.camera_pivot.world_position
            camera.world_rotation_y = self.world_rotation_y

    def check_respawn(self):
        if held_keys["g"]:
            self._record_buffer = []
            self.reset_car()
        if held_keys["v"]:
            if self.multiray_sensor:
                self.multiray_sensor.set_enabled_rays(
                    not self.multiray_sensor.enabled
                )
        if self.y <= -100:
            self.reset_car()
        if self.y >= 300:
            self.reset_car()


    def cap_kinetic_parameters(self):
        if self.speed >= self.topspeed:
            self.speed = self.topspeed
        if self.speed <= -15:
            self.speed = -15
        if self.speed <= 0:
            self.pivot.rotation_y = self.rotation_y

        if self.rotation_speed >= self.max_rotation_speed:
            self.rotation_speed = self.max_rotation_speed
        if self.rotation_speed <= -self.max_rotation_speed:
            self.rotation_speed = -self.max_rotation_speed

        if self.camera_rotation >= 40:
            self.camera_rotation = 40
        elif self.camera_rotation <= 30:
            self.camera_rotation = 30

    def update_vertical_position(self, y_ray, movementY):
        if self.visible:
            if y_ray.distance <= self.scale_y * 1.7 + abs(movementY):
                self.velocity_y = 0
                if (
                    y_ray.world_normal.y > 0.7
                    and y_ray.world_point.y - self.world_y < 0.5
                ):
                    self.y = y_ray.world_point.y + 1.4
                    self.hitting_wall = False
                else:
                    self.hitting_wall = True

                if self.copy_normals:
                    self.ground_normal = self.position + y_ray.world_normal
                else:
                    self.ground_normal = self.position + (0, 180, 0)
            else:
                self.y += movementY * 50 * time.dt
                self.velocity_y -= 50 * time.dt

    # ===== Snapshot builder (utilisé pour autopilot + recording) =====
    def _build_snapshot(self, controls):
        """
        controls: tuple(bool forward, bool back, bool left, bool right)
        """
        snap = SensingSnapshot()
        snap.current_controls = tuple(bool(c) for c in controls)

        snap.car_position = tuple(self.world_position)
        snap.car_speed = float(self.speed)
        snap.car_angle = float(self.rotation_y)

        # Raycasts
        if self.multiray_sensor:
            try:
                snap.raycast_distances = list(
                    self.multiray_sensor.collect_sensor_values()
                )
            except Exception:
                snap.raycast_distances = [0.0]
        else:
            snap.raycast_distances = [0.0]

        # Image (screen capture)
        try:
            tex = base.win.getDisplayRegion(0).getScreenshot()
            arr = tex.getRamImageAs("RGB")
            img = np.frombuffer(arr, np.uint8).reshape(
                tex.getYSize(), tex.getXSize(), 3
            )
            snap.image = img[::-1]  # flip Y
        except Exception:
            snap.image = None

        snap.timestamp = time.time()
        return snap

    def _save_record_buffer(self):
        if not self._record_buffer:
            print("[REC] No data to save!")
            return

        record_name = os.path.join(self._record_dir, "record_%d.npz")
        fid = 0
        while os.path.exists(record_name % fid):
            fid += 1
        path = record_name % fid

        print(f"[REC] Saving {len(self._record_buffer)} snapshots to {path}...")
        try:
            with lzma.open(path, "wb") as f:
                pickle.dump(self._record_buffer, f)
            print(f"[REC] Successfully saved to {path}")
        except Exception as e:
            print(f"[REC] Save failed: {e}")
        finally:
            self._record_buffer = []

    # ===== Core update =====
    def update(self):
        # dt fixe comme dans l'original
        time.dt = 1 / 40

        # Respawn & divers
        self.check_respawn()

        # 1) Récupérer les inputs courants (humain ou autopilot)
        forward, back, left, right = self.input_manager.get_inputs()

        # 2) Construire un snapshot à partir de l'état + action courante
        snap = self._build_snapshot((forward, back, left, right))

        # 3) Autopilot : mettre à jour les commandes pour la frame suivante
        if self.input_manager.mode == InputMode.AUTOPILOT:
            controls_next = self.autopilot.process_snapshot(snap)
            if controls_next:
                self.input_manager.set_autopilot_output(*controls_next)

        # 4) Physique d'origine, en utilisant forward/back/left/right

        # Accélération / friction
        if forward:
            self.speed += self.acceleration * time.dt
            self.driving = True
        else:
            self.driving = False
            if self.speed > 1:
                self.speed -= self.friction * 5 * time.dt
            elif self.speed < -1:
                self.speed += self.friction * 5 * time.dt

        # Frein
        if back:
            if self.speed > 0:
                self.speed -= self.braking_strenth * time.dt
            else:
                self.speed -= self.acceleration * time.dt
            self.braking = True
        else:
            self.braking = False

        # Contraintes de vitesse
        if self.speed > self.topspeed:
            self.speed = self.topspeed
        elif hasattr(self, "minspeed") and self.speed < self.minspeed:
            self.speed = self.minspeed
        elif not hasattr(self, "minspeed") and self.speed < -15:
            self.speed = -15
        
        if left or right:
            turn_right = right and not left
            rotation_sign = (1 if turn_right else -1)

            normalized_speed = abs(self.speed / self.topspeed)

            def rotation_radius(norm_speed):
                smallest_radius = 1.5
                biggest_radius = 25
                return pow(norm_speed, 1.5) * (biggest_radius - smallest_radius) + smallest_radius

            radius = rotation_radius(normalized_speed)
            travelled_dist = abs(self.speed * time.dt)
            travelled_circle_center_angle = travelled_dist / radius
            dx = 1 - cos(travelled_circle_center_angle)
            dy = sin(travelled_circle_center_angle)
            da = atan2(dx, dy) / 3.14159 * 180
            self.rotation_y += da * rotation_sign

        # ===== COLLISION HANDLING du document 2 (sans abs()) =====
        total_dist_to_move = self.speed * time.dt

        def move_car(distance_to_travel, direction):
            front_collision = boxcast(
                origin=self.world_position,
                direction=self.forward * direction,
                thickness=(0.1, 0.1),
                distance=self.scale_x + distance_to_travel,  # SANS abs()
                ignore=[self],
            )

            if front_collision.distance < self.scale_x + distance_to_travel:
                free_dist = front_collision.distance - self.scale_x + distance_to_travel
                next_forward = self.forward - (
                    self.forward.dot(front_collision.world_normal)
                ) * front_collision.world_normal
                self.speed = self.speed * (
                    0.5 + 0.5 * (self.forward.dot(front_collision.world_normal))
                )
                self.rotation_y = (
                    atan2(next_forward[0], next_forward[2]) / 3.14159 * 180
                )

                OBSTACLE_DISPLACEMENT_MARGIN = 1
                self.x += (front_collision.world_normal * OBSTACLE_DISPLACEMENT_MARGIN).x
                self.z += (front_collision.world_normal * OBSTACLE_DISPLACEMENT_MARGIN).z
                return 0
            else:
                self.x += self.forward[0] * distance_to_travel
                self.z += self.forward[2] * distance_to_travel
                return 0

        # On fait 2 itérations comme dans le code d'origine
        for _ in range(2):
            total_dist_to_move = move_car(
                total_dist_to_move, 1 if self.speed > 0 else -1
            )
            if total_dist_to_move <= 0:
                break

        # Mise à jour caméras / pivots
        self.c_pivot.position = self.position
        self.c_pivot.rotation_y = self.rotation_y
        self.update_camera()
        self.pivot.position = self.position

        # 5) Recording : on garde TOUTES les frames si activé
        if self._record_enabled:
            self._record_buffer.append(snap)

    def input(self, key):
        # Toggle autopilot
        if key == "t":
            self.input_manager.toggle_autopilot()

        # Toggle recording
        if key == "r":
            self._record_enabled = not self._record_enabled
            print("[REC] =", self._record_enabled)

        # ESC : flush record puis sortie
        if key == "escape":
            if self._record_enabled:
                print("[REC] Flushing buffer...")
                self._save_record_buffer()
            print("Exiting game ...")
            os._exit(0)

    def reset_car(self):
        self.position = self.reset_position
        self.rotation_y = self.reset_orientation[1]

        camera.world_rotation_y = self.rotation_y
        self.speed = 0
        self.velocity_y = 0
        self.timer_running = False

        # reset trails
        for trail in self.trails:
            if trail.trailing:
                trail.end_trail()
        self.start_trail = True

        # reset LSTM/autopilot
        try:
            self.autopilot.engine.reset()
        except Exception:
            pass


    def update_model_path(self):
        self.model_path = str(self.model).replace("render/scene/car/", "")
        invoke(self.update_model_path, delay=3)


class CarRepresentation(Entity):
    def __init__(self, car, position=(0, 0, 0), rotation=(0, 65, 0)):
        super().__init__(
            parent=scene,
            model="assets/cars/sports-car.obj",
            texture="assets/cars/garage/sports-car/sports-red.png",
            position=position,
            rotation=rotation,
            scale=(1, 1, 1),
        )
        self.model_path = str(self.model).replace(
            "render/scene/car_representation/", ""
        )
        self.text_object = None
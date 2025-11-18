from ursina import *
import json
from direct.stdpy import thread
from pathlib import Path
import json


def load_track_metadata(track_name):
    """
    Loads metadata for a given track.

    Args:
        track_name (str): The name of the track (which corresponds to the subfolder name in assets).

    Returns:
        dict: The metadata for the track.
    """
    # Construct the full package path for the track's metadata
    root_dir = Path(__file__).resolve().parent.parent
    asset_path = root_dir / f"assets/{track_name}/track_metadata.json"
    with open(asset_path, "r") as f:
        metadata = json.load(f)

    return metadata


class Track(Entity):
    def __init__(self, track_name):

        self.track_name = track_name
        self.checkpoints = []       # Will hold the invisible checkpoint Entities
        self.current_checkpoint = 0 # The index of the next checkpoint to hit
        
        self.data = load_track_metadata(track_name)

        # Find assets paths.
        track_model_path =  str(self.data["track_model"])
        track_texture_path = str(self.data["track_texture"])
        
        origin_position = tuple(self.data["origin_position"])
        _origin_rotation_data = tuple(self.data["origin_rotation"]) # Load into a temp variable
        self.origin_scale = tuple(self.data["origin_scale"])

        self.car_default_reset_position = tuple(self.data["car_default_reset_position"])
        _car_default_reset_orientation_data = tuple(self.data["car_default_reset_orientation"]) # Load into a temp variable

        # --- THIS IS THE FIX ---
        # Apply -90 rotation to both the track's rotation AND the car's default rotation
        origin_rotation = (_origin_rotation_data[0], _origin_rotation_data[1] - 90, _origin_rotation_data[2])
        self.car_default_reset_orientation = (_car_default_reset_orientation_data[0], _car_default_reset_orientation_data[1] - 90, _car_default_reset_orientation_data[2])

        finish_line_position = tuple(self.data["finish_line_position"])
        finish_line_rotation = tuple(self.data["finish_line_rotation"])
        finish_line_scale = tuple(self.data["finish_line_scale"])
       
        print("Creating track entity")
        super().__init__(model = track_model_path, texture = load_texture(track_texture_path),
                         position = origin_position, rotation = origin_rotation, 
                         scale = self.origin_scale, collider = "mesh")
        print("Done creating track entity")

        self.load_checkpoints()

        self.finish_line = Entity(model = "cube", position = finish_line_position,
                                  rotation = finish_line_rotation, scale = finish_line_scale, visible = False)
        self.track = [ self.finish_line ]

        self.details = []
        for detail in self.data["details"]:
            self.details.append(Entity(model = detail["model"], texture = load_texture(detail["texture"]),
                            position = origin_position, rotation_y = origin_rotation[1], 
                            scale = self.origin_scale[1]))
        self.obstacles = []
        for obstacle in self.data["obstacles"]:
            self.obstacles.append(Entity(model = obstacle["model"],
                            collider = "mesh",
                            position = origin_position, rotation_y = origin_rotation[1], 
                            scale = self.origin_scale[1], visible = False))

        self.disable()
        
        self.played = False
        self.unlocked = False

        self.deactivate()
    
    def load_checkpoints(self):
        """
        Loads checkpoint data from a JSON file and creates invisible
        trigger entities in the world.
        """
        # Assumes assets folder is at the project root
        checkpoint_file = f'assets/{self.track_name}/SimpleTrack_checkpoints.json'
        print(f"Attempting to load checkpoints from: {checkpoint_file}")
        
        try:
            with open(checkpoint_file) as f:
                data = json.load(f)
                
                for cp in data['checkpoints']:
                    # 1. Get the raw coordinates
                    raw_x, raw_y, raw_z = cp['position']

                    # 2. NEGATE X and Z to rotate 180 degrees
                    # We map (x, y, z) -> (-x, y+5, -z)
                    adjusted_pos = (-raw_x, raw_z + 5, -raw_y)
                    # 3. Create the Entity at the new position
                    e = Entity(
                        parent = self, # <-- Keep this fix from before!
                        model = 'quad',
                        scale = (40, 20),
                        position = adjusted_pos, # <-- Use the swapped position
                        collider = 'box',
                        visible = False,
                        color = color.red,
                        texture = 'white_cube',
                        double_sided = True
                    )
                    # e.animate_rotation_y(360, duration=5, loop=True) 
                    self.checkpoints.append(e)
                    
            print(f"Successfully loaded {len(self.checkpoints)} checkpoints.")

        except FileNotFoundError:
            print(f"WARNING: No checkpoint file found at {checkpoint_file}. GA progress will not work.")
        except Exception as e:
            print(f"ERROR loading checkpoints: {e}")

    def deactivate(self):
        for i in self.track:
            i.disable()
        for i in self.details:
            i.disable()
        for i in self.obstacles:
            i.disable()
        self.disable()

    def activate(self, activate_details = True):
        self.enable()
        for i in self.track:
            i.enable()
        for i in self.obstacles:
            i.enable()
        if activate_details:
            for i in self.details:
                i.enable()


    def load_assets(self, global_models = [], global_texs = []):
        def inner_load_assets():
            models_to_load = list(set(
                                      [detail["model"] for detail in self.data["details"]] +
                                      [obs["model"] for obs in self.data["obstacles"]]))

            textures_to_load = list(set(
                                        [detail["texture"] for detail in self.data["details"]] +
                                        [obs["texture"] for obs in self.data["obstacles"]]
                                        ))

            # Load global models and textures.
            for i, m in enumerate(global_models):
                print("Loading global model")
                print(m)
                load_model(m)
            for i, t in enumerate(global_texs):
                print("Loading global texture")
                print(t)
                load_texture(t)
                
            for i, m in enumerate(models_to_load):
                print("Loading local model")
                print(m)
                model_path = str(m)
                load_model(model_path)

            for i, t in enumerate(textures_to_load):
                print("Loading local texture")
                print(t)
                texture_path = str(t)
                load_texture(texture_path)

        try:
            thread.start_new_thread(function=inner_load_assets, args="")
        except Exception as e:
            print("error starting thread", e)

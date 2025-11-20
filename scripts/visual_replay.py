import sys
import os
import json
from ursina import *
from ursina.prefabs.dropdown_menu import DropdownMenu

# --- 1. PATH SETUP ---
# We need to help Python find the 'rallyrobopilot' package
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(scripts_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# CRITICAL: Ursina uses the Current Working Directory to find assets.
# We force it to the project root.
os.chdir(project_root)
print(f"📂 Working Directory set to: {os.getcwd()}")

# --- 2. IMPORT THE WORKING LAUNCHER ---
try:
    from rallyrobopilot.game_launcher import prepare_game_app
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("   Ensure you have downloaded all .py files (including the rallyrobopilot folder).")
    sys.exit()

# --- 3. INITIALIZE GAME (EXACTLY LIKE MAIN.PY) ---
print("🚀 Initializing Game via game_launcher...")
# This function loads the window, track, car, and ASSETS correctly.
app, car = prepare_game_app()

# --- 4. CUSTOMIZE FOR REPLAY ---
# We can modify the app settings after initialization
window.title = "RRP - Live Spectator"
window.borderless = False
window.size = (800, 800) # Vertical strip

# Move Camera to 'Spectator View'
camera.parent = car.c_pivot
camera.position = (0, 60, -60)
camera.rotation = (45, 0, 0)
camera.fov = 90

# --- 4.5 RENDER CHECKPOINTS (NEW) ---
visible_checkpoints = []
try:
    with open("assets/SimpleTrackCP/SimpleTrack_checkpoints.json", 'r') as f:
        checkpoint_data = json.load(f)
    
    if 'checkpoints' in checkpoint_data:
        total_checkpoints = len(checkpoint_data['checkpoints'])
        # Apply the same transformation as in track.py: (-x, z+5, -y)
        positions = [(-cp['position'][0], cp['position'][2] + 5, -cp['position'][1]) for cp in checkpoint_data['checkpoints']]

        for i, pos in enumerate(positions):
            # Calculate a progressive hue for a rainbow effect
            hue = (i / total_checkpoints) * 360
            checkpoint_color = color.hsv(hue, 1, 1, a=0.5) # Added transparency
            
            cp_entity = Entity(
                parent=car.track,
                model='cube',
                position=pos,
                scale=(40, 10, 5), # Width, Height, Depth
                rotation=(0, 0, 0), # Default rotation for now
                color=checkpoint_color,
                name=f"checkpoint_{i+1}"
            )
            visible_checkpoints.append(cp_entity)
        print(f"✅ Rendered {len(visible_checkpoints)} checkpoints with default rotation.")
except Exception as e:
    print(f"⚠️ Could not render checkpoints: {e}")

# --- 5. REPLAY LOGIC (CONTROLLED EXTERNALLY) ---
COMMAND_FILE = "replay_command.txt"
LAST_COMMAND_CHECK = 0
CHECK_INTERVAL = 0.25  # Check for new commands 4 times a second

# --- State Variables ---
replay_data = {}
replay_actions = []
current_step = 0
is_playing = False
is_paused = True # Start in a paused state
last_lap_progress = 0

# --- HUD ---
hud_text = Text(
    text="Waiting for command from controller...",
    position=window.top_left,
    origin=(-0.5, 0.5),
    scale=1.2,
    background=True
)

def update_hud():
    """Updates the HUD text with the current replay status."""
    if not replay_data:
        hud_text.text = "Waiting for command from controller..."
        return

    status = "PAUSED" if is_paused else "PLAYING"
    gen = replay_data.get('generation', 'N/A')
    fit = replay_data.get('fitness', 0)
    vel = replay_data.get('velocity', 0)
    total_steps = len(replay_actions)
    
    hud_text.text = (
        f"Status: {status} | Step: {current_step}/{total_steps}\n"
        f"Gen: {gen} | Fitness: {fit:.0f} | Avg Vel: {vel:.1f}"
    )

def load_replay(filepath):
    """Loads a new replay and resets the simulation state."""
    global replay_data, replay_actions, current_step, is_playing, is_paused, last_lap_progress
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                replay_data = json.load(f)
            
            replay_actions = replay_data.get('genome', [])
            print(f"Loaded {os.path.basename(filepath)} ({len(replay_actions)} steps)")
            
            car.reset_car()
            current_step = 0
            is_playing = True
            is_paused = False # Always start playing on load

            # Reset checkpoint colors and progress
            last_lap_progress = 0
            total_checkpoints = len(visible_checkpoints)
            if total_checkpoints > 0:
                for i, cp_entity in enumerate(visible_checkpoints):
                    hue = (i / total_checkpoints) * 360
                    cp_entity.color = color.hsv(hue, 1, 1, a=0.5)
        else:
            print(f"Error: File not found at {filepath}")
            replay_data = {"error": "File not found"}
            
    except Exception as e:
        print(f"Read Error: {e}")
        replay_data = {"error": "Error loading file"}

def check_for_command():
    """Checks the command file for new commands and executes them."""
    global LAST_COMMAND_CHECK, is_paused
    if time.time() - LAST_COMMAND_CHECK < CHECK_INTERVAL:
        return
    LAST_COMMAND_CHECK = time.time()
    
    try:
        if not os.path.exists(COMMAND_FILE):
            return
            
        with open(COMMAND_FILE, 'r') as f:
            command = f.read().strip()
        
        if command:
            # Clear the command file immediately after reading
            with open(COMMAND_FILE, 'w') as f:
                f.write('')
            
            if command == "toggle_pause":
                if is_playing:
                    is_paused = not is_paused
                    print(f"Playback {'paused' if is_paused else 'resumed'}")
            else: # It's a filepath
                print(f"Received command to load: {command}")
                load_replay(command)
    except Exception as e:
        print(f"Error checking command file: {e}")

# --- 6. GAME LOOP ---
def update():
    """Ursina update loop, called every frame."""
    global current_step, last_lap_progress
    
    check_for_command()
    update_hud()
    
    if is_playing and not is_paused and current_step < len(replay_actions):
        action = replay_actions[current_step]
        
        # Apply recorded controls
        held_keys['w'] = 'forward' in action
        held_keys['s'] = 'back' in action
        held_keys['a'] = 'left' in action
        held_keys['d'] = 'right' in action
        
        # Run simulation step
        time.dt = 1/40.0
        car.update()
        
        # --- NEW: Checkpoint Color Change ---
        if car.lap_progress > last_lap_progress:
            # A new checkpoint was hit
            hit_checkpoint_index = car.lap_progress - 1
            if 0 <= hit_checkpoint_index < len(visible_checkpoints):
                visible_checkpoints[hit_checkpoint_index].color = color.lime
            last_lap_progress = car.lap_progress
        
        current_step += 1
        
    elif is_playing and not is_paused and current_step >= len(replay_actions):
        # Auto-loop when finished
        car.reset_car()
        current_step = 0
        last_lap_progress = 0 # Reset progress
        # Reset colors
        total_checkpoints = len(visible_checkpoints)
        if total_checkpoints > 0:
            for i, cp_entity in enumerate(visible_checkpoints):
                hue = (i / total_checkpoints) * 360
                cp_entity.color = color.hsv(hue, 1, 1, a=0.5)

# The application now starts and waits for commands.
app.run()





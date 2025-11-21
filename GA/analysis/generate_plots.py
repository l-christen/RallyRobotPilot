import os
import sys
import matplotlib.pyplot as plt
import time

# --- 1. SET UP PATHS ---
# Add project root to the path to allow importing 'rallyrobopilot'
current_script_path = os.path.abspath(__file__)
analysis_dir = os.path.dirname(current_script_path)
ga_dir = os.path.dirname(analysis_dir)
project_root = os.path.dirname(ga_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)
print(f"✅ Project Root set to: {os.getcwd()}")

# --- 2. IMPORTS & HEADLESS APP SETUP ---
from ursina import *
from rallyrobopilot.game_launcher import prepare_game_app

print("🚀 Initializing Headless Ursina App...")
# Initialize the game, but we won't run the main loop
app, car = prepare_game_app()
window.enabled = False # Disable the game window

# --- 3. DEFINE REPRESENTATIVE GENOMES ---
# These are hand-crafted genomes to represent "slow" vs. "fast" driving styles
# for the first segment of the track.

# Represents the "slow and safe" model: cautious, wide turns.
GENOME_SLOW = (
    ['push forward;'] * 25 +
    ['push forward; push left;'] * 40 +
    ['push forward;'] * 25
)

# Represents the "fast and aggressive" model: straighter lines, higher speed.
GENOME_FAST = (
    ['push forward;'] * 50 +
    ['push left;'] * 5 +
    ['push forward;'] * 20 +
    ['push forward; push right;'] * 15
)

# --- 4. SIMULATION FUNCTION ---
def run_simulation(genome):
    """
    Runs a headless simulation for a given genome and records the trajectory.
    """
    print(f"    Simulating genome with {len(genome)} actions...")
    car.reset_car()
    
    trajectory = []
    speeds = []
    
    # Store initial state
    trajectory.append((car.world_x, car.world_z))
    speeds.append(car.speed)

    for action in genome:
        # Apply controls from the genome
        held_keys['w'] = 'forward' in action
        held_keys['s'] = 'back' in action
        held_keys['a'] = 'left' in action
        held_keys['d'] = 'right' in action
        
        # Manually step the simulation
        car.update()
        
        # Record results
        trajectory.append((car.world_x, car.world_z))
        speeds.append(car.speed)
        
        # Clear keys for the next step
        held_keys['w'] = held_keys['s'] = held_keys['a'] = held_keys['d'] = 0

    print("    Simulation complete.")
    return trajectory, speeds

# --- 5. RUN SIMULATIONS ---
print("\n--- Running Simulation for SLOW Model ---")
slow_traj, slow_speeds = run_simulation(GENOME_SLOW)

print("\n--- Running Simulation for FAST Model ---")
fast_traj, fast_speeds = run_simulation(GENOME_FAST)

# --- 6. GENERATE PLOTS ---
print("\n🎨 Generating comparison plots...")

# PLOT 1: Trajectory Comparison
plt.figure(figsize=(10, 8))
slow_x, slow_z = zip(*slow_traj)
fast_x, fast_z = zip(*fast_traj)
plt.plot(slow_x, slow_z, label='Model 1: "Slow & Safe" Trajectory', color='blue', linestyle='--')
plt.plot(fast_x, fast_z, label='Model 2: "Aggressive & Fast" Trajectory', color='red')
plt.title('GA Model Trajectory Comparison (First Segment)')
plt.xlabel('X Coordinate')
plt.ylabel('Z Coordinate')
plt.legend()
plt.grid(True)
plt.gca().invert_xaxis() # Match game's coordinate system if necessary
plt.axis('equal')
trajectory_plot_path = os.path.join(analysis_dir, 'trajectory_comparison.png')
plt.savefig(trajectory_plot_path)
print(f"✅ Saved Trajectory plot to: {trajectory_plot_path}")

# PLOT 2: Speed Profile Comparison
plt.figure(figsize=(10, 6))
plt.plot(slow_speeds, label='Model 1: "Slow & Safe" Speed', color='blue', linestyle='--')
plt.plot(fast_speeds, label='Model 2: "Aggressive & Fast" Speed', color='red')
plt.title('GA Model Speed Profile Comparison')
plt.xlabel('Simulation Step')
plt.ylabel('Car Speed (m/s)')
plt.legend()
plt.grid(True)
speed_plot_path = os.path.join(analysis_dir, 'speed_profile_comparison.png')
plt.savefig(speed_plot_path)
print(f"✅ Saved Speed Profile plot to: {speed_plot_path}")

# --- 7. CLEANUP ---
print("\n✅ Visualization generation complete.")
app.destroy()
sys.exit()

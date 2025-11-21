import os
import sys
import json
import glob
import random
import matplotlib.pyplot as plt
import numpy as np

# --- 1. SET UP PATHS ---
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
app, car = prepare_game_app()
window.enabled = False

# --- 3. CONFIGURATION ---
REPLAYS_DIR = 'replays/'
SAMPLE_SIZE = 50 # Number of replay files to analyze
if not os.path.isdir(REPLAYS_DIR):
    print(f"❌ Error: Replays directory '{REPLAYS_DIR}' not found.")
    sys.exit()

# --- 4. LOAD REPLAY FILES ---
all_replay_files = glob.glob(os.path.join(REPLAYS_DIR, 'replay_*.json'))
if not all_replay_files:
    print(f"❌ Error: No 'replay_*.json' files found in '{REPLAYS_DIR}'.")
    sys.exit()

# Take a random sample if there are more files than SAMPLE_SIZE
if len(all_replay_files) > SAMPLE_SIZE:
    replay_files_sample = random.sample(all_replay_files, SAMPLE_SIZE)
else:
    replay_files_sample = all_replay_files
print(f"✅ Found {len(all_replay_files)} replay files. Analyzing a sample of {len(replay_files_sample)}.")

# --- 5. SIMULATION FUNCTION ---
def run_simulation_from_genome(genome):
    car.reset_car()
    trajectory = []
    
    for action in genome:
        held_keys['w'] = 'forward' in action
        held_keys['s'] = 'back' in action
        held_keys['a'] = 'left' in action
        held_keys['d'] = 'right' in action
        
        car.update()
        
        trajectory.append((car.world_x, car.world_z))
        held_keys['w'] = held_keys['s'] = held_keys['a'] = held_keys['d'] = 0
        
    return trajectory

# --- 6. PROCESS REPLAYS AND GENERATE PLOTS ---
all_trajectories = []
all_fitness_scores = []
all_velocities = []

print("\nProcessing replay files...")
for i, file_path in enumerate(replay_files_sample):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
            all_fitness_scores.append(data.get('fitness', 0))
            all_velocities.append(data.get('velocity', 0))
            
            if 'genome' in data:
                print(f"  ({i+1}/{len(replay_files_sample)}) Simulating {os.path.basename(file_path)}...")
                traj = run_simulation_from_genome(data['genome'])
                all_trajectories.append(traj)
    except Exception as e:
        print(f"⚠️ Could not process file {file_path}: {e}")

print("\n🎨 Generating visualization plots...")

# PLOT 1: Trajectory "Spaghetti" Plot
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 8))
for i, traj in enumerate(all_trajectories):
    x, z = zip(*traj)
    plt.plot(x, z, color='royalblue', alpha=0.3, linewidth=1)
plt.gca().invert_xaxis()
plt.title(f'Failed Run Trajectories (Sample of {len(all_trajectories)} Genomes)')
plt.xlabel('X Coordinate')
plt.ylabel('Z Coordinate')
plt.axis('equal')
plt.grid(True)
trajectory_plot_path = os.path.join(analysis_dir, 'failed_run_trajectories.png')
plt.savefig(trajectory_plot_path)
print(f"✅ Saved Trajectory plot to: {trajectory_plot_path}")

# PLOT 2: Fitness Distribution
plt.figure(figsize=(10, 6))
plt.hist(all_fitness_scores, bins=20, color='mediumpurple', edgecolor='black')
plt.title('Distribution of Fitness Scores in Failed Runs')
plt.xlabel('Final Fitness Score')
plt.ylabel('Number of Genomes')
plt.grid(True)
fitness_plot_path = os.path.join(analysis_dir, 'failed_run_fitness_distribution.png')
plt.savefig(fitness_plot_path)
print(f"✅ Saved Fitness Distribution plot to: {fitness_plot_path}")

# PLOT 3: Velocity Distribution
plt.figure(figsize=(10, 6))
plt.hist(all_velocities, bins=20, color='coral', edgecolor='black')
plt.axvline(x=0, color='k', linestyle='--', linewidth=1, label='Stopped')
plt.title('Distribution of Final Velocities in Failed Runs')
plt.xlabel('Final Velocity (m/s)')
plt.ylabel('Number of Genomes')
plt.legend()
plt.grid(True)
velocity_plot_path = os.path.join(analysis_dir, 'failed_run_velocity_distribution.png')
plt.savefig(velocity_plot_path)
print(f"✅ Saved Velocity Distribution plot to: {velocity_plot_path}")

# --- 7. CLEANUP ---
print("\n✅ Analysis complete.")
app.destroy()
sys.exit()

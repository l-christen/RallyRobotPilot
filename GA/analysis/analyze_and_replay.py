import requests
import time
import json
import csv
import sys
import matplotlib
from datetime import datetime  # <--- NEW IMPORT

# --- Force Headless Mode (Must be before importing pyplot) ---
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
WORKER_URL = "http://localhost:5001"
GENOME_FILE = "/home/jeremy.duc/nas_home/RallyRobotPilot/GA/seed/human_seed.json" 
DELTA_T = 0.05 

def load_genome(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {filename}")
        sys.exit(1)

def sanitize_command(cmd):
    if not cmd or cmd.strip() == "":
        return "push forward;" 
    return cmd

def run_analysis():
    # --- 1. Setup (Timestamped Output) ---
    # Generate a timestamp string (Year-Month-Day_Hour-Minute-Second)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Define the base directory
    output_dir = Path("GA/telemetry")
    output_dir.mkdir(parents=True, exist_ok=True) # Create folders if missing
    
    # Create dynamic filenames
    csv_file = output_dir / f"best_genome_analysis_{timestamp}.csv"
    png_file = output_dir / f"best_genome_analysis_{timestamp}.png"

    print(f"Outputs will be saved to:\n -> {csv_file}\n -> {png_file}")

    genome = load_genome(GENOME_FILE)
    print(f"Loaded Genome: {len(genome)} steps")
    
    try:
        requests.post(f"{WORKER_URL}/command", json={'command': 'reset;'})
        time.sleep(1.0) 
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {WORKER_URL}. Is the simulator running?")
        return

    telemetry = []
    x_coords = []
    y_coords = [] 

    print("Starting Replay...")
    start_time = time.time()

    # 2. The Physics Loop
    for i, raw_cmd in enumerate(genome):
        step_start = time.time()
        cmd = sanitize_command(raw_cmd)

        try:
            resp = requests.get(f"{WORKER_URL}/sensing", timeout=0.2)
            state = resp.json()
        except Exception as e:
            print(f"Sensing failed: {e}")
            break

        cx = state.get('car_position x', 0)
        cz = state.get('car_position z', 0) 
        
        telemetry.append({
            'step': i,
            'time': i * DELTA_T,
            'cmd': cmd,
            'x': cx,
            'z': cz,
            'speed': state.get('car_speed', 0),
            'collided': state.get('collided', False)
        })
        
        x_coords.append(cx)
        y_coords.append(cz) 

        requests.post(f"{WORKER_URL}/command", json={'command': cmd})

        elapsed = time.time() - step_start
        sleep_time = DELTA_T - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
            
        if i % 20 == 0:
            sys.stdout.write(f"\rStep {i}/{len(genome)} | Speed: {state.get('car_speed', 0):.2f}")
            sys.stdout.flush()

    # 3. Save Data (Using dynamic CSV filename)
    print(f"\nSaving telemetry to {csv_file}...")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=telemetry[0].keys())
        writer.writeheader()
        writer.writerows(telemetry)

    # 4. Visualization (Using dynamic PNG filename)
    print(f"Generating Trajectory Plot to {png_file}...")
    plt.figure(figsize=(10, 8))
    plt.plot(x_coords, y_coords, label='GA Trajectory', color='blue', linewidth=2)
    
    if x_coords: # Avoid error if empty
        plt.scatter(x_coords[0], y_coords[0], color='green', s=100, label='Start')
        plt.scatter(x_coords[-1], y_coords[-1], color='red', marker='X', s=100, label='End')
    
    plt.title(f"Genome Analysis: {timestamp}")
    plt.xlabel("Position X")
    plt.ylabel("Position Z (Forward/Depth)")
    plt.axis('equal') 
    plt.grid(True)
    plt.legend()
    
    plt.savefig(png_file)
    print("✅ Plot saved successfully.")

if __name__ == "__main__":
    run_analysis()
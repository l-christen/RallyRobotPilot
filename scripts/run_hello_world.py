import requests
import time

# --- Configuration ---
DELTA_T = 0.1  # Must match the 0.1s (10Hz) we discussed
GENOME_DURATION_SEC = 10
GENOME_LENGTH = int(GENOME_DURATION_SEC / DELTA_T)

# --- A "smarter" genome to test the turn ---
# 1. Go straight for 3 seconds (30 steps) to build speed
genome_straight = ['push forward;'] * 30

# 2. Turn right for 2 seconds (20 steps)
# We use both commands, separated by a space, as defined in remote_commands.py
genome_turn_right = ['push forward; push right;'] * 20

# 3. Go straight again for 5 seconds (50 steps)
genome_rest = ['push forward;'] * 50

# Combine them into our new test genome
hello_world_genome = genome_straight + genome_turn_right + genome_rest
GENOME_LENGTH = len(hello_world_genome) # Update the length

# This is the address of your Docker container
BASE_URL = "http://localhost:5000"

# --- Main "Hello World" Function ---
def run_hello_world():
    """
    Connects to the simulation, runs a single hard-coded genome,
    and reports the final fitness score.
    """
    print("--- Starting 'Hello World' GA Client ---")
    
    # --- 1. Reset the simulation ---
    print(f"Sending 'reset' command to {BASE_URL}/command...")
    try:
        requests.post(f"{BASE_URL}/command", json={'command': 'reset;'})
    except requests.exceptions.ConnectionError as e:
        print("\n--- !! CONNECTION FAILED !! ---")
        print(f"Could not connect to {BASE_URL}.")
        print("Is your Docker container running?")
        print("Run 'docker-compose up headless' in your other terminal.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during reset: {e}")
        return

    print("Reset successful. Starting genome execution...")
    print("-" * 60)
    print(f"  {'Step':<5} | {'Time':<7} | {'Progress':<10} | {'Speed':<7} | {'Position (x, z)':<20}")
    print("-" * 60)

    # --- 2. Run the genome ---
    for i, action in enumerate(hello_world_genome):
        # Send the action (e.g., 'push forward;')
        try:
            requests.post(f"{BASE_URL}/command", json={'command': action})
        except Exception as e:
            print(f"Error sending command: {e}")
            break # Stop if we lose connection

        # Wait for the fixed time step
        time.sleep(DELTA_T)
        
        # --- NEW DEBUGGING LOGIC ---
        # Get the state of the car AFTER the action
        try:
            response = requests.get(f"{BASE_URL}/sensing")
            data = response.json()
            
            # Extract key data for logging
            progress = data.get('lap_progress', 0)
            time_now = data.get('current_lap_time', 0.0)
            speed = data.get('car_speed', 0.0)
            x_pos = data.get('car_position x', 0.0)
            z_pos = data.get('car_position z', 0.0)
            
            # Format a new, rich log message (f-strings align the columns)
            print(f"  {i+1:03d}/{GENOME_LENGTH} | "
                  f"{time_now:6.2f}s | "
                  f"{progress:<10d} | "
                  f"{speed:6.1f} | "
                  f"({x_pos:6.1f}, {z_pos:6.1f})")

        except Exception as e:
            # If sensing fails, print a simpler message so we know
            print(f"  Step {i+1:03d}/{GENOME_LENGTH}: Sent action '{action}' (Sensing failed: {e})")
        # --- END NEW DEBUGGING LOGIC ---

    print("\n--- Genome execution complete ---")

    # --- 3. Get the final score ---
    # We still get the data one last time to be sure
    print("Requesting final sensing data...")
    try:
        response = requests.get(f"{BASE_URL}/sensing")
        data = response.json()

        progress = data.get('lap_progress', 0)
        time_taken = data.get('current_lap_time', 999.0)
        laps_completed = data.get('laps', 0)

        # Calculate fitness
        fitness = (laps_completed * 5000) + (progress * 100) - time_taken

        # --- 4. Print the final report ---
        print("\n--- FINAL FITNESS REPORT ---")
        print(f"Laps Completed:     {laps_completed}")
        print(f"Checkpoints Reached: {progress}")
        print(f"Final Time:         {time_taken:.2f}s")
        print("------------------------------")
        print(f"FITNESS SCORE:      {fitness:.2f}")

    except Exception as e:
        print(f"Error getting final sensing data: {e}")


# --- This makes the script runnable ---
if __name__ == "__main__":
    run_hello_world()
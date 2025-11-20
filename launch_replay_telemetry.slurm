#!/bin/bash
#SBATCH --job-name=rally_replay
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --partition=Calypso
#SBATCH --output=logs/replay_%j.log   # Redirects output to a log file
#SBATCH --error=logs/replay_%j.err

# --- 0. Environment Setup ---
# Adjust these lines to match how you installed dependencies on the cluster
module load python  # Example module load
# source ~/rally_env/bin/activate # Example venv activation

# --- 1. Clean start ---
echo "Cleaning up previous workers..."
./manage_workers.sh stop 1

# --- 2. Start 1 worker (Port 5001) ---
echo "Starting worker..."
./manage_workers.sh start 1
echo "Waiting 15s for worker to initialize..."
sleep 15 # Increased to 15s to be safe

# --- 3. Run the recorder ---
echo "Starting Telemetry Recorder..."
# Ensure the script saves the image instead of showing it!
python3 GA/analysis/analyze_and_replay.py

# --- 4. Cleanup ---
echo "Stopping worker..."
./manage_workers.sh stop 1
echo "Job Complete."
#!/bin/bash
#SBATCH --job-name=rally_seg
#SBATCH --output=seg_job_%j.out
#SBATCH --error=seg_job_%j.err
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=10
#SBATCH --exclude=calypso1,calypso9
#SBATCH --time=08:00:00
#SBATCH --partition=Calypso

cd /home/jeremy.duc/nas_home/RallyRobotPilot/
source venv/bin/activate

# --- PHASE 1: SMART DISTRIBUTION ---
echo "📦 Checking Docker images on nodes..."
# CRITICAL FIX: Only run 1 task per node for Docker loading to prevent corruption
srun --nodes=6 --ntasks=6 --ntasks-per-node=1 \
     bash -c "sg docker -c 'docker image inspect rallyrobopilot-headless > /dev/null 2>&1 || docker load -i /home/jeremy.duc/nas_home/RallyRobotPilot/rally_headless.tar'"

echo "✅ Images ready. Starting MPI Segmented Run..."

# --- PHASE 2: EXECUTION ---
mpirun \
    --mca btl tcp,self \
    --mca btl_tcp_if_include 192.168.91.0/24 \
    --mca oob_tcp_if_include 192.168.91.0/24 \
    --mca pml ob1 \
    python3 -u GA/scripts/run_ga_segmented.py
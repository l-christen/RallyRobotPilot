# launch_mpi.sh

#!/bin/bash
#SBATCH --job-name=rally_mpi
#SBATCH --output=mpi_job_%j.out
#SBATCH --error=mpi_job_%j.err
#SBATCH --nodes=8            
#SBATCH --ntasks-per-node=5  
#SBATCH --exclude=calypso1,calypso9
#SBATCH --time=04:00:00
#SBATCH --partition=Calypso

cd /home/jeremy.duc/nas_home/RallyRobotPilot/
source venv/bin/activate

# --- PHASE 1: SMART DISTRIBUTION ---
echo "📦 Checking Docker images on nodes..."
srun --nodes=8 --ntasks=8 --ntasks-per-node=1 \
     bash -c "sg docker -c 'docker image inspect rallyrobopilot-headless > /dev/null 2>&1 || docker load -i /home/jeremy.duc/nas_home/RallyRobotPilot/rally_headless.tar'"

echo "✅ Images ready. Starting MPI..."

# --- PHASE 2: EXECUTION ---
# THE FIX:
# 1. --mca btl_tcp_if_include 192.168.91.0/24 : Force Data traffic onto the physical wire
# 2. --mca oob_tcp_if_include 192.168.91.0/24 : Force Manager traffic onto the physical wire
# 3. -u : Force Python to print logs immediately (Unbuffered)

mpirun \
    --mca btl tcp,self \
    --mca btl_tcp_if_include 192.168.91.0/24 \
    --mca oob_tcp_if_include 192.168.91.0/24 \
    --mca pml ob1 \
    python3 -u scripts/run_ga_mpi.py
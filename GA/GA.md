# How to Launch the Segmented GA Pipeline (Cluster/MPI)

This process runs the Genetic Algorithm on the cluster using SLURM and MPI. Unlike local testing, the Python script automatically manages the Docker containers on the worker nodes.

### Prerequisites
Ensure the following files are in place:
* `scripts/run_ga_segmented.py` (The Master/Worker logic)
* `launch_segmented.sh` (The SLURM submission script)

---

## 1. Prepare the Environment

Connect to the cluster and navigate to your project root.

```bash
# 1. Go to your project folder
cd ~/RallyRobotPilot/

# 2. Activate your Python Virtual Environment
source venv/bin/activate
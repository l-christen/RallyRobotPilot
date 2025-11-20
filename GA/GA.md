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
```

---

## 2. How the Pipeline Works

The pipeline is designed to find an optimal driving path for a given track using a distributed Genetic Algorithm.

1.  **Launch**: The process is initiated by the `launch_segmented.sh` script, which submits a job to the SLURM workload manager. SLURM allocates the requested nodes on the cluster.

2.  **MPI Execution**: The SLURM job executes `run_ga_segmented.py` using MPI (`mpirun`). This starts multiple instances of the Python script across the allocated nodes.
    *   **Master Node (Rank 0)**: The first instance acts as the master. It orchestrates the entire GA process, manages the population of genomes, and assigns tasks to workers.
    *   **Worker Nodes (Rank > 0)**: All other instances act as workers. Each worker starts a headless Docker container running the car simulation. Their job is to evaluate the genomes sent by the master.

3.  **Segmented Evolution**: The GA doesn't try to solve the whole track at once. Instead, it breaks the track down into segments (from one checkpoint to the next) and solves them sequentially. For each segment:
    *   The master initializes a population of "genomes" (which are sequences of driving actions).
    *   It sends these genomes to the workers for evaluation.
    *   Workers simulate the genome and report back a fitness score, which is heavily weighted towards achieving a high exit velocity.
    *   The master uses these scores to perform selection, crossover, and mutation, creating a new generation of genomes.
    *   This loop continues until a genome is found that meets the segment's exit velocity requirement.

4.  **Output Generation**: As the GA runs, it produces two main types of output files in the `GA/results_segmented/` directory:
    *   `live_spectator.json`: This file is updated every generation with the current best genome. It can be used to monitor the GA's performance in real-time.
    *   `genome_segment_XX.json`: Upon the successful completion of each segment, a detailed JSON file is saved. It contains a rich set of metadata, including the final segment genome, GA parameters, performance metrics (fitness, velocity), and a full trajectory recording. These files are designed to be captured by an external script (e.g., a `.bat` script on a Windows machine) for analysis and replay.

5.  **Visualization**: Once the run is complete, the generated `genome_segment_XX.json` files can be loaded by the `scripts/visual_replay.py` script. This allows you to visually replay the optimal path found by the GA. During the replay, checkpoints will change color to green when the car successfully passes through them.
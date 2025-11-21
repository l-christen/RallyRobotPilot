# Genetic Algorithm (GA) Pipeline Documentation

This document provides an overview of the segmented Genetic Algorithm pipeline used to find optimal driving trajectories for the Rally Robot Pilot.

## 1. How to Launch the Segmented GA Pipeline (Cluster/MPI)

This process runs the Genetic Algorithm on the cluster using SLURM and MPI. Unlike local testing, the Python script automatically manages the Docker containers on the worker nodes.

### Prerequisites
Ensure the following files are in place:
* `scripts/run_ga_segmented.py` (The Master/Worker logic)
* `launch_segmented.sh` (The SLURM submission script)

### Step 1: Prepare the Environment

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

---

## 3. Technical Deep Dive: The Genetic Algorithm

This section details the theory, implementation, and evolution of the project's Genetic Algorithm.

### 3.1. The Goal: Defining "Optimal Trajectory"

The primary objective of the GA is to find an "optimal trajectory" for the car. In this context, optimality is not merely about completing the track, but about doing so as quickly as possible.

-   **Optimization Problem**: The task is to find an ideal sequence of discrete driving actions (a "genome") that successfully navigates the car through a series of checkpoints.
-   **The Objective Function**: Optimality is quantified by a **fitness function**. This function evaluates each genome and assigns it a score. The GA's goal is to maximize this score. The key insight of this implementation is that a fast lap time is the result of maintaining high speed. Therefore, the fitness function is designed to aggressively reward the **exit velocity** at the end of each track segment. A high exit velocity from one segment is crucial for a good entry speed into the next, creating a virtuous cycle that leads to a faster overall lap time.

### 3.2. Technical Implementation & Evolution

-   **Core Framework**: The GA is built using Python with `mpi4py` for massively parallel processing across a high-performance computing cluster. It uses Docker to create isolated, headless environments for each simulation, ensuring that evaluations are consistent and do not interfere with each other. This parallelism is the cornerstone of the GA, as it allows for the evaluation of thousands of potential solutions in a reasonable timeframe.

-   **Evolution of the Implementation**:
    1.  **"Hello World" Pipeline**: The project began by establishing a basic, end-to-end communication pipeline between a master node and worker nodes to prove the viability of the distributed architecture.
    2.  **Robust Parallel Framework**: The core parallel GA logic was then built out, implementing the master/worker task distribution for evaluating genomes.
    3.  **Determinism & Seeding**: A critical milestone was achieving deterministic physics by locking the simulation's time step (`DELTA_T`). A GA requires that a given genome produces the exact same outcome every time; without this, fitness scores are meaningless. This commit also introduced **human seeding**, allowing the GA to be initialized with a known-good (though not necessarily fast) solution to accelerate the search for a valid path.
    4.  **Segmented Approach**: The final architecture uses a segmented approach, optimizing the track checkpoint-by-checkpoint. This modularizes the problem, making it more manageable than attempting to optimize a full lap genome from scratch.

### 3.3. Analysis of Failures & Path to Improvement

-   **The Velocity Barrier (The Core Problem)**: The primary reason for the GA failing to complete a full run has been its inability to achieve the required exit velocity (e.g., 15.0 m/s) for each segment. The logs consistently show that while the GA is excellent at finding *a path* to the next checkpoint, it learns to do so slowly and cautiously.

-   **Why Does This Happen? Analysis of Past Failures**:
    1.  **Local Optima**: The GA repeatedly gets trapped in a "local optimum." It discovers that slow, safe paths are a reliable way to get the large fitness bonus for reaching a checkpoint. It then spends all its effort making minor tweaks to these slow paths rather than exploring radically different, higher-risk, higher-reward (i.e., faster) paths.
    2.  **Over-reliance on a Slow Seed**: The initial human-provided seed, while effective at finding a valid path, was likely very slow. This biases the entire initial population towards slowness, effectively "poisoning the well" and making it harder for fast solutions to emerge.
    3.  **Premature Convergence**: The original adaptive mutation schedule decreased the exploration rate too quickly. The GA would "converge" on a slow solution early on and then stop exploring for better alternatives.

-   **Recent Improvements (Implemented in commit `c8f46de`)**: To address these failures, the following improvements were implemented:
    *   **Increased Velocity Reward**: The fitness function was retuned to more aggressively reward speed. The velocity multiplier was increased from `50.0` to `150.0`, making high-speed solutions significantly more attractive to the algorithm.
    *   **Enhanced Exploration**: To break out of local optima, the GA was forced to explore more. The initial mutation rate was increased from `0.10` to `0.15`, and its rate of decay was slowed down, ensuring that the GA continues to search for novel solutions for a longer portion of the run.

-   **Further Recommendations for Improvement**:
    *   **Continued Fitness Function Tuning**: If the velocity problem persists, the balance of rewards and penalties needs further tuning. One could introduce a penalty for taking too many steps (i.e., too much time) to complete a segment, explicitly punishing slowness.
    *   **Advanced GA Techniques**: Consider implementing more advanced techniques like a "Hall of Fame," which preserves the absolute best individuals found across *all* generations, preventing them from being lost. More sophisticated crossover and mutation operators could also be explored.
    *   **Seeding Strategy**: Experiment with different seeding strategies. Instead of using a single human seed, one could use a mix of several different human paths (some safe, some risky) or reduce the influence of the seed by injecting more purely random individuals into the initial population.
    *   **Reviewing the Action Space**: The current set of discrete `POSSIBLE_ACTIONS` may be too limited to permit the fine-grained control needed for high-speed cornering. Future work could explore a more continuous control scheme, although this would significantly increase the complexity of the genome.
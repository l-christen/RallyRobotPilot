# Understanding the MPI Implementation

This document explains how the Message Passing Interface (MPI) is used in this project to parallelize the Genetic Algorithm and how to modify its key parameters.

## 1. How MPI Works in This Project

The project uses a **Master/Worker** model to distribute the workload of the Genetic Algorithm. This is implemented in the `GA/scripts/run_ga_segmented.py` script using the `mpi4py` library.

### 1.1. Core Concepts

-   **Communicator**: All processes are part of a group called a "communicator" (`MPI.COMM_WORLD`).
-   **Rank**: Each process in the group is assigned a unique number called a "rank". The process with `rank == 0` is designated as the **Master**, and all other processes (`rank > 0`) are **Workers**.

### 1.2. The Master/Worker Communication Flow

The entire process is orchestrated by the Master node.

1.  **Initialization**:
    *   The Master (rank 0) initializes the Genetic Algorithm, sets up the population of genomes, and identifies the current track segment to be optimized.
    *   Each Worker (rank > 0) starts and immediately launches its own sandboxed simulation environment inside a dedicated Docker container. The script is designed so each worker uses a different port to avoid conflicts.

2.  **Work Distribution (Master -> Workers)**:
    *   The Master iterates through the population of genomes that need to be evaluated.
    *   It sends a work packet to each available Worker using `comm.send(data, dest=worker_rank, tag=100)`.
    *   The `data` packet contains the genome to be tested, along with contextual information like the historical actions and the target checkpoint.
    *   The `tag=100` is a message identifier that tells the Worker, "This is a genome for you to evaluate."

3.  **Evaluation (Workers)**:
    *   The Worker waits to receive a message.
    *   Upon receiving a message with `tag=100`, it unpacks the genome and runs the simulation by sending commands to its local Docker container.
    *   After the simulation is complete, it calculates the fitness score based on the car's performance (final velocity, distance to target, etc.).

4.  **Result Collection (Workers -> Master)**:
    *   The Worker sends the result (a tuple containing `fitness`, `progress`, `velocity`) back to the Master using `comm.send(result, dest=0, tag=200)`.
    *   The `tag=200` tells the Master, "This is the result of the evaluation you asked for."

5.  **Synchronization and Next Generation**:
    *   The Master collects results from all Workers.
    *   While waiting, if a Worker finishes its task, the Master immediately sends it the next available genome from the population, ensuring the workers are always busy.
    *   Once all results for a generation are collected, the Master performs selection, crossover, and mutation to create the next generation of the population.
    *   The entire cycle (steps 2-5) repeats for the specified number of generations.

### 1.3. Control Flow & Special Tags

MPI tags are used to manage the state of the application:
-   `tag=100`: A work packet (genome evaluation) from Master to Worker.
-   `tag=200`: A result packet (fitness score) from Worker to Master.
-   `tag=300`: A special request from the Master to a single Worker to record and return a full trajectory for the best-found genome.
-   `tag=666`: A shutdown signal from the Master to all Workers, telling them to terminate their loops and close their Docker containers.
-   `tag=0`: An idle/wait signal. The Master sends this when there is no more work to distribute for the current generation, telling the worker to wait.

## 2. How to Change GA Parameters

If you are asked to change the Genetic Algorithm's parameters during your presentation, you can do so by editing the configuration variables at the top of the `GA/scripts/run_ga_segmented.py` file.

Here are the key parameters and how to modify them:

| Parameter Name      | Purpose                                                                                                 | Example: How to Change                                                                      |
| ------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `POPULATION_SIZE`   | The number of individuals (genomes) in each generation. A larger population explores more but is slower.  | `POPULATION_SIZE = 100` (Increases the number of parallel evaluations per generation).      |
| `NUM_GENERATIONS`   | The number of generations the GA will run for each track segment before concluding.                         | `NUM_GENERATIONS = 30` (Allows the GA more "time" to evolve a better solution per segment). |
| `NUM_ELITES`        | The number of the very best individuals from one generation that are guaranteed to survive into the next. | `NUM_ELITES = 2` (Reduces elitism, which can sometimes increase diversity).               |
| `MUTATION_RATE`     | The initial probability (e.g., `0.15` = 15%) that a part of a genome will be randomly changed (mutated).   | `MUTATION_RATE = 0.25` (Significantly increases randomness and exploration).                |
    | `SEGMENT_STEPS`     | The number of actions in a genome, defining the "horizon" of the simulation for each segment.           | `SEGMENT_STEPS = 150` (Gives the car more actions/time to reach the next checkpoint).       |

To change a parameter, simply open the `run_ga_segmented.py` file, find the variable in the `# --- CONFIGURATION ---` section, and change its value.

### 2.1. Changing MPI/SLURM Execution Parameters (in 'launch_segmented.sh')

The `launch_segmented.sh` script is a SLURM (Simple Linux Utility for Resource Management) job script used to submit and manage the parallel execution of the GA on a computing cluster. It defines the resources requested and how the MPI processes are launched.

You can modify the following key parameters within `launch_segmented.sh` to control the MPI execution:

| Parameter Name        | SLURM Directive          | Purpose                                                                                                 | Example: How to Change                                              |
| --------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| **Job Name**          | `#SBATCH --job-name`     | A descriptive name for your job, useful for identifying it in the SLURM queue.                          | `#SBATCH --job-name=my_new_ga_run`                                 |
| **Number of Nodes**   | `#SBATCH --nodes`        | The total number of computing nodes you are requesting for your job.                                    | `#SBATCH --nodes=10` (Requests 10 nodes for the job).               |
| **Tasks per Node**    | `#SBATCH --ntasks-per-node` | The number of MPI processes (or "tasks") that will be launched on each allocated node.                  | `#SBATCH --ntasks-per-node=5` (Runs 5 MPI processes on each node).  |
| **Total MPI Tasks**   | (Implicit)               | The total number of MPI processes that will run is `nodes * ntasks-per-node`. In this script, each MPI task corresponds to a GA worker. | If `nodes=6` and `ntasks-per-node=10`, then `6 * 10 = 60` total MPI tasks. |
| **Time Limit**        | `#SBATCH --time`         | The maximum time duration your job is allowed to run (e.g., `HH:MM:SS`).                                | `#SBATCH --time=10:00:00` (Sets a 10-hour time limit).              |
| **Exclude Nodes**     | `#SBATCH --exclude`      | Comma-separated list of specific nodes to exclude from the job allocation.                              | `#SBATCH --exclude=calypso5,calypso6`                               |
| **Partition**         | `#SBATCH --partition`    | The SLURM partition (queue) to submit the job to. This might vary depending on your cluster's setup.    | `#SBATCH --partition=MyDedicatedPartition`                         |

**Example Modification:**

To run the GA with 120 workers distributed across 12 nodes (10 tasks per node) for a maximum of 4 hours, you would modify `launch_segmented.sh` as follows:

```bash
#!/bin/bash
#SBATCH --job-name=rally_seg_120workers # Changed job name
#SBATCH --output=seg_job_%j.out
#SBATCH --error=seg_job_%j.err
#SBATCH --nodes=12                   # Changed from 6 to 12 nodes
#SBATCH --ntasks-per-node=10         # Kept at 10 tasks per node
#SBATCH --exclude=calypso1,calypso9
#SBATCH --time=04:00:00              # Changed from 08:00:00 to 04:00:00
#SBATCH --partition=Calypso
# ... rest of the script ...
```

By adjusting these parameters, you can scale the GA's execution to leverage more or fewer resources on your computing cluster, directly impacting the speed and thoroughness of the evolutionary process.

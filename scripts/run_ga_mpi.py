from mpi4py import MPI
import requests
import time
import random
import json
import os
import subprocess
import sys
import math
import datetime

# --- PLOTTING SETUP (HEADLESS) ---
import matplotlib
# Force matplotlib to not use any Xwindow backend to prevent crashes on headless nodes
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
GENOME_LENGTH = 500
POPULATION_SIZE = 32
NUM_GENERATIONS = 20
NUM_ELITES = 4
MUTATION_RATE = 0.02
DELTA_T = 1/40.0  # Locked to 40 FPS physics tick

# NAS PATHS
NAS_HOME = "/home/jeremy.duc/nas_home/RallyRobotPilot/GA"
SEED_FILE = os.path.join(NAS_HOME, "seed/human_seed.json")
RESULTS_DIR = os.path.join(NAS_HOME, "results")

POSSIBLE_ACTIONS = [
    'push forward;',
    'push forward; push right;',
    'push forward; push left;',
    'push right;',
    'push left;',
    ''
]

# --- ASSET LOADING ---
def load_checkpoints():
    """Loads checkpoints from the assets folder relative to this script."""
    try:
        # Navigate up from scripts/ to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoint_file = os.path.join(project_root, 'assets/SimpleTrackCP/SimpleTrack_checkpoints.json')
        
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
            return data['checkpoints']
    except Exception as e:
        print(f"⚠️ Error loading checkpoints: {e}")
        return []

CHECKPOINTS = load_checkpoints()

def load_human_seed():
    """Loads the human seed from the NAS."""
    try:
        if os.path.exists(SEED_FILE):
            print(f"🧬 Loading seed from: {SEED_FILE}")
            with open(SEED_FILE, "r") as f:
                return json.load(f)
        else:
            print(f"⚠️ No seed file found at {SEED_FILE}")
            return None
    except Exception as e:
        print(f"⚠️ Error reading seed: {e}")
        return None

# --- WORKER: DOCKER MANAGEMENT ---

def run_docker_cmd(rank, cmd_list):
    """
    Runs a Docker command wrapped in 'sg docker' to ensure group permissions.
    Using a list prevents shell injection and quoting issues.
    """
    # Construct the inner command string
    inner_cmd = " ".join(cmd_list)
    
    # Wrap in sg docker -c "..."
    final_cmd = ["sg", "docker", "-c", inner_cmd]
    
    try:
        # Capture stderr to debug errors if they happen
        subprocess.run(final_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        err_str = e.stderr.decode().strip()
        # Ignore common cleanup errors (container not found)
        if "No such container" in err_str or "No such process" in err_str:
            return False
            
        print(f"[Rank {rank}] 🛑 Docker Error: {err_str}")
        print(f"[Rank {rank}] ⚠️ Failed Command: {final_cmd}")
        return False

def start_docker_container(rank):
    """Starts a headless container on the LOCAL node for this specific rank."""
    port = 5000 + rank 
    container_name = f"worker_{rank}_rally"
    
    # 1. Force cleanup of old containers
    run_docker_cmd(rank, ["docker", "rm", "-f", container_name])
    
    # 2. Start Command
    # We use the image name 'rallyrobopilot-headless' that we loaded earlier
    start_args = [
        "docker", "run", "-d", "--rm",
        "--name", container_name,
        "-p", f"{port}:5000",
        "rallyrobopilot-headless"
    ]
    
    if run_docker_cmd(rank, start_args):
        time.sleep(5) # Warmup time for Flask/Ursina
        return f"http://localhost:{port}", container_name
    else:
        return None, None

def stop_docker_container(container_name):
    if container_name:
        run_docker_cmd(0, ["docker", "kill", container_name])

# --- SIMULATION ---
def evaluate_genome(genome, base_url):
    """Runs the simulation for one genome and calculates fitness."""
    if not base_url:
        return -99999, 0

    try:
        # 1. Reset Simulation
        requests.post(f"{base_url}/command", json={'command': 'reset;'}, timeout=1)
        
        # 2. Execute Genome
        for action in genome:
            requests.post(f"{base_url}/command", json={'command': action}, timeout=0.5)
            time.sleep(DELTA_T)

        # 3. Sensing & Fitness Calc
        response = requests.get(f"{base_url}/sensing", timeout=1)
        data = response.json()

        progress = data.get('lap_progress', 0)
        laps_completed = data.get('laps', 0)
        final_x = data.get('car_position x', 0.0)
        final_z = data.get('car_position z', 0.0)

        # Base Fitness
        fitness = (laps_completed * 5000) + (progress * 100)
        
        # Time Penalty / Efficiency
        if laps_completed > 0:
            fitness -= data.get('current_lap_time', 999.0)
        else:
            # Penalize taking too long to do nothing
            fitness -= len(genome) * DELTA_T

        # Distance Bonus (Corrected for Coordinate System)
        dist_bonus = 0.0
        if progress < len(CHECKPOINTS):
            next_cp = CHECKPOINTS[progress]
            raw_pos = next_cp['position'] # [x, y, z] from JSON
            
            # COORDINATE TRANSFORM: 
            # track.py spawns entities at (-raw_x, ..., -raw_y)
            target_x = -raw_pos[0]
            target_z = -raw_pos[1] 
            
            # Calculate distance in Game World coordinates
            dist = math.dist((final_x, final_z), (target_x, target_z))
            
            # Bonus: Inverse distance (closer = higher score)
            dist_bonus = (1.0 / (dist + 1.0)) * 1000 
            fitness += dist_bonus

        return fitness, progress

    except Exception as e:
        # If worker fails mid-run (e.g., timeout), return bad score
        return -99999, 0

# --- GA OPERATIONS ---
def create_random_genome():
    return [random.choice(POSSIBLE_ACTIONS) for _ in range(GENOME_LENGTH)]

def mutate(genome):
    mutated_genome = []
    for gene in genome:
        if random.random() < MUTATION_RATE:
            mutated_genome.append(random.choice(POSSIBLE_ACTIONS))
        else:
            mutated_genome.append(gene)
    return mutated_genome

def crossover(parent1, parent2):
    if GENOME_LENGTH <= 2: return parent1
    pt = random.randint(1, GENOME_LENGTH - 2)
    return parent1[:pt] + parent2[pt:]

def create_initial_population():
    seed = load_human_seed()
    population = []
    
    if seed and len(seed) > 0:
        # Truncate or pad seed
        if len(seed) > GENOME_LENGTH:
            seed = seed[:GENOME_LENGTH]
        else:
            seed += [''] * (GENOME_LENGTH - len(seed))
            
        print(f"🧬 Seeding population with Human Genome.")
        population.append(seed) # Elitism for the seed
        
        # Local Search: Fill rest with mutations of the seed
        for _ in range(POPULATION_SIZE - 1):
            population.append(mutate(seed))
    else:
        print("🎲 No valid seed found. Using random population.")
        population = [create_random_genome() for _ in range(POPULATION_SIZE)]
        
    return population

# --- MAIN MPI CONTROLLER ---
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        print("Error: Need at least 2 MPI processes (1 Master, 1 Worker)")
        sys.exit(1)

    # ---------------------------------------
    # MASTER NODE (Rank 0)
    # ---------------------------------------
    if rank == 0:
        print(f"👑 Master started. Managing {size-1} workers.")
        print(f"📂 Saving results to: {RESULTS_DIR}")
        os.makedirs(RESULTS_DIR, exist_ok=True)

        history_best = []
        history_avg = []
        
        population = create_initial_population()
        start_time = time.time()

        for gen in range(NUM_GENERATIONS):
            print(f"\n--- 🏁 GENERATION {gen+1}/{NUM_GENERATIONS} ---")
            
            fitness_results = [] # Stores (score, genome)
            active_jobs = {}     # Map {worker_rank: genome_being_processed}
            next_genome_idx = 0
            
            # 1. Initial Send (Fill all workers)
            for w in range(1, size):
                if next_genome_idx < len(population):
                    genome_to_send = population[next_genome_idx]
                    comm.send(genome_to_send, dest=w, tag=100)
                    active_jobs[w] = genome_to_send
                    next_genome_idx += 1
                else:
                    comm.send(None, dest=w, tag=0) # Idle wait

            # 2. Collect & Redistribute Loop
            while len(fitness_results) < len(population):
                status = MPI.Status()
                # Receive result
                score, progress = comm.recv(source=MPI.ANY_SOURCE, tag=200, status=status)
                sender = status.Get_source()
                
                # Retrieve the genome associated with this worker
                processed_genome = active_jobs.pop(sender)
                fitness_results.append((score, processed_genome))
                
                # Send next job if available
                if next_genome_idx < len(population):
                    genome_to_send = population[next_genome_idx]
                    comm.send(genome_to_send, dest=sender, tag=100)
                    active_jobs[sender] = genome_to_send
                    next_genome_idx += 1
                
            # 3. Statistics
            fitness_only = [r[0] for r in fitness_results]
            current_best = max(fitness_only)
            current_avg = sum(fitness_only) / len(fitness_only)
            
            history_best.append(current_best)
            history_avg.append(current_avg)
            
            print(f"  📊 Stats: Best={current_best:.2f}, Avg={current_avg:.2f}")

            # 4. Evolution (Elitism + Crossover/Mutation)
            fitness_results.sort(key=lambda x: x[0], reverse=True)
            new_pop = []
            
            # Elitism
            elites = [g for s, g in fitness_results[:NUM_ELITES]]
            new_pop.extend(elites)
            
            # Breeding
            while len(new_pop) < POPULATION_SIZE:
                # Tournament Selection
                candidates1 = random.sample(fitness_results[:16], 2)
                p1 = max(candidates1, key=lambda x: x[0])[1]
                
                candidates2 = random.sample(fitness_results[:16], 2)
                p2 = max(candidates2, key=lambda x: x[0])[1]
                
                child = crossover(p1, p2)
                child = mutate(child)
                new_pop.append(child)
                
            population = new_pop

        # --- END OF GA ---
        print(f"\n🏆 GA Complete in {(time.time() - start_time)/60:.1f} mins.")
        
        # Save Best Genome
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        final_best_genome = fitness_results[0][1]
        genome_filename = f"{RESULTS_DIR}/best_genome_{timestamp}.json"
        with open(genome_filename, "w") as f:
            json.dump(final_best_genome, f, indent=2)
        print(f"✅ Best genome saved to {genome_filename}")

        # Generate Plot
        plt.figure(figsize=(10, 6))
        gens = range(1, NUM_GENERATIONS + 1)
        plt.plot(gens, history_best, label='Best Fitness', color='green', linewidth=2)
        plt.plot(gens, history_avg, label='Average Fitness', color='blue', linestyle='--', linewidth=2)
        plt.title(f"GA Evolution ({POPULATION_SIZE} Indiv, {size-1} Workers)")
        plt.xlabel("Generation")
        plt.ylabel("Fitness Score")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_filename = f"{RESULTS_DIR}/fitness_graph_{timestamp}.png"
        plt.savefig(plot_filename)
        print(f"📈 Graph saved to {plot_filename}")

        # Kill Workers
        for w in range(1, size):
            comm.send(None, dest=w, tag=666) 

    # ---------------------------------------
    # WORKER NODES (Rank > 0)
    # ---------------------------------------
    else:
        # 1. Staggered Startup (Prevents resource storms)
        startup_delay = (rank % 10) * 2
        time.sleep(startup_delay)

        # 2. Start Docker
        base_url, container_name = start_docker_container(rank)
        
        if not base_url:
            print(f"[Rank {rank}] Failed to start Docker. Zombie mode.")
        
        # 3. Event Loop
        try:
            while True:
                status = MPI.Status()
                genome = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
                tag = status.Get_tag()

                if tag == 666: # Kill Signal
                    break
                
                if tag == 0: # Sleep/Idle
                    time.sleep(0.1)
                    continue

                if genome is not None:
                    if base_url:
                        fitness, progress = evaluate_genome(genome, base_url)
                    else:
                        fitness, progress = -99999, 0 
                        
                    comm.send((fitness, progress), dest=0, tag=200)
        
        except Exception as e:
            print(f"[Rank {rank}] Critical Error: {e}")
        
        finally:
            # 4. Cleanup
            stop_docker_container(container_name)

if __name__ == "__main__":
    main()
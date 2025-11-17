import requests
import time
import random
import os
import datetime
import json
import math # NEW: Import math library
from multiprocessing import Pool, cpu_count

# --- 🚀 PARALLEL CONFIGURATION ---
NUM_WORKERS = 32
BASE_PORT = 5000
WORKER_URLS = [f"http://localhost:{BASE_PORT + i}" for i in range(1, NUM_WORKERS + 1)]

# --- GA Configuration ---
DELTA_T = 0.1

# --- NEW: Load Checkpoint Data ---
def load_checkpoints():
    """
    Loads the checkpoint data from the JSON file.
    Assumes this script is in the 'scripts' folder.
    """
    # Go up one level (from 'scripts' to project root)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_file = os.path.join(project_root, 'assets/SimpleTrackCP/SimpleTrack_checkpoints.json')
    
    try:
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
            # Store just the list of checkpoints
            return data['checkpoints']
    except Exception as e:
        print(f"FATAL ERROR: Could not load {checkpoint_file}. Exiting.")
        print(f"Error: {e}")
        exit()

# Load checkpoints ONCE when the script starts
CHECKPOINTS = load_checkpoints()
print(f"Successfully loaded {len(CHECKPOINTS)} checkpoints.")

# --- MODIFIED: calculate_fitness now returns a dict ---
def calculate_fitness(genome: list[str], base_url: str) -> dict:
    """
    Runs a single genome and returns a dictionary of its results.
    """
    
    # --- 1. Reset the simulation ---
    try:
        requests.post(f"{base_url}/command", json={'command': 'reset;'})
    except Exception as e:
        return {'score': -99999, 'progress': 0, 'laps': 0, 'final_time': 0, 'dist_bonus': 0}

    # --- 2. Run the genome ---
    for action in genome:
        try:
            requests.post(f"{base_url}/command", json={'command': action})
        except Exception as e:
            break 
        time.sleep(DELTA_T)

    # --- 3. Get the final score ---
    try:
        response = requests.get(f"{base_url}/sensing")
        data = response.json()

        progress = data.get('lap_progress', 0)
        laps_completed = data.get('laps', 0)
        
        final_x = data.get('car_position x', 0.0)
        final_z = data.get('car_position z', 0.0)
        
        # --- !! NEW FITNESS CALCULATION !! ---
        
        # 1. Base score for laps and checkpoints
        fitness = (laps_completed * 5000) + (progress * 100)
        
        # 2. Time penalty
        if laps_completed > 0:
            time_taken = data.get('current_lap_time', 999.0)
            fitness -= time_taken
            final_time = time_taken
        else:
            time_penalty = len(genome) * DELTA_T
            fitness -= time_penalty
            final_time = time_penalty

        # 3. NEW: Distance Bonus
        # 'progress' is the index of the *next* checkpoint to hit
        distance_bonus = 0.0
        if progress < len(CHECKPOINTS):
            next_cp = CHECKPOINTS[progress]
            
            # Get 2D positions
            car_pos_2d = (final_x, final_z)
            # Checkpoint positions are [x, y, z]
            cp_pos_2d = (next_cp['position'][0], next_cp['position'][2]) 
            
            # Calculate 2D distance
            distance = math.dist(car_pos_2d, cp_pos_2d)
            
            # Bonus is inversely proportional to distance.
            # We scale it by 1000 to make it significant.
            # We add 1e-6 to avoid division by zero.
            distance_bonus = (1.0 / (distance + 1e-6)) * 1000 
            
            fitness += distance_bonus

        # Return the full result object
        return {
            'score': fitness,
            'progress': progress,
            'laps': laps_completed,
            'final_time': final_time,
            'dist_bonus': distance_bonus
        }

    except Exception as e:
        print(f"Error getting final sensing from {base_url}: {e}")
        return {'score': -99999, 'progress': 0, 'laps': 0, 'final_time': 0, 'dist_bonus': 0}

# --- (Rest of the script is identical to your last version) ---

# --- GA initialization ---
POSSIBLE_ACTIONS = [
    'push forward;',
    'push forward; push right;',
    'push forward; push left;',
    'push right;',
    'push left;',
    ''
]

GENOME_LENGTH = 400     # 40-second run
POPULATION_SIZE = 32  # SMOKE TEST: One individual per worker
NUM_GENERATIONS = 10  # SMOKE TEST: 10 generations
NUM_ELITES = 4
MUTATION_RATE = 0.05

def create_random_genome() -> list[str]:
    return [random.choice(POSSIBLE_ACTIONS) for _ in range(GENOME_LENGTH)]

def create_initial_population() -> list[list[str]]:
    return [create_random_genome() for _ in range(POPULATION_SIZE)]

def select_parents(fitness_scores: list) -> (list, list):
    top_half = fitness_scores[:POPULATION_SIZE // 2]
    if not top_half:
        parent1 = fitness_scores[0][1]
        parent2 = fitness_scores[0][1]
        return parent1, parent2
    parent1 = random.choice(top_half)[1]
    parent2 = random.choice(top_half)[1]
    while parent1 == parent2 and len(top_half) > 1:
        parent2 = random.choice(top_half)[1]
    return parent1, parent2

def crossover(parent1: list, parent2: list) -> list:
    if GENOME_LENGTH <= 2: return parent1
    crossover_point = random.randint(1, GENOME_LENGTH - 2)
    child_genome = parent1[:crossover_point] + parent2[crossover_point:]
    return child_genome

def mutate(genome: list) -> list:
    mutated_genome = []
    for gene in genome:
        if random.random() < MUTATION_RATE:
            mutated_genome.append(random.choice(POSSIBLE_ACTIONS))
        else:
            mutated_genome.append(gene)
    return mutated_genome

# --- MODIFIED: Main GA Execution (to log the bonus) ---
def run_ga():
    print("Creating initial population...")
    population = create_initial_population()

    pool = Pool(processes=NUM_WORKERS)
    print(f"Worker pool with {NUM_WORKERS} processes created.")

    for gen in range(NUM_GENERATIONS):
        print(f"\n--- 🏁 GENERATION {gen+1}/{NUM_GENERATIONS} ---")
        
        print(f"Evaluating {len(population)} individuals on {NUM_WORKERS} workers...")
        
        tasks = []
        for i, genome in enumerate(population):
            worker_url = WORKER_URLS[i % NUM_WORKERS]
            tasks.append((genome, worker_url)) 
            
        async_results = [pool.apply_async(calculate_fitness, t) for t in tasks]
        
        fitness_scores = []
        all_results_data = []
        
        for i, async_result in enumerate(async_results):
            result_data = async_result.get() 
            all_results_data.append(result_data)
            
            # --- NEW LIVE LOG ---
            score = result_data['score']
            progress = result_data['progress']
            bonus = result_data['dist_bonus']
            print(f"  Indiv {i+1:02d}/{POPULATION_SIZE}: Score={score:6.2f}, Progress={progress}, DistBonus={bonus:5.2f}")

            fitness_scores.append((score, population[i]))
        
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        best_result_this_gen = max(all_results_data, key=lambda r: r['score'])
        
        print("\n--- Generation Summary ---")
        print(f"Best Fitness: {best_result_this_gen['score']:.2f} (Progress: {best_result_this_gen['progress']}, DistBonus: {best_result_this_gen['dist_bonus']:.2f})")
        
        avg_score = sum(r['score'] for r in all_results_data) / POPULATION_SIZE
        print(f"Avg Fitness:  {avg_score:.2f}")

        new_population = []
        elites = [genome for score, genome in fitness_scores[:NUM_ELITES]]
        new_population.extend(elites)
        
        while len(new_population) < POPULATION_SIZE:
            p1, p2 = select_parents(fitness_scores)
            child = crossover(p1, p2)
            mutated_child = mutate(child)
            new_population.append(mutated_child)
            
        population = new_population

    # --- GA Finished ---
    print("\n" + "=" * 30)
    print("--- 🏆 GA Complete! ---")
    
    print(f"Running final evaluation on {len(population)} individuals...")
    final_tasks = [pool.apply_async(calculate_fitness, (population[i], WORKER_URLS[i % NUM_WORKERS])) for i in range(len(population))]
    
    final_fitness_scores = []
    final_results_data = []
    
    for i, async_result in enumerate(final_tasks):
        result_data = async_result.get()
        final_results_data.append(result_data)
        final_fitness_scores.append((result_data['score'], population[i]))
        
    final_fitness_scores.sort(key=lambda x: x[0], reverse=True)
    best_final_result = max(final_results_data, key=lambda r: r['score'])
    best_genome = final_fitness_scores[0][1]
    
    print(f"Best ever fitness: {best_final_result['score']:.2f}")
    print(f"  (Progress: {best_final_result['progress']}, Final Time: {best_final_result['final_time']:.2f}s, DistBonus: {best_final_result['dist_bonus']:.2f})")
    
    print("\nBest genome (first 10 actions):")
    print(best_genome[:10])
    
    output_dir = "GA_results"
    os.makedirs(output_dir, exist_ok=True)
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{output_dir}/best_genome_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(best_genome, f, indent=2)
        
    print(f"\n✅ Best genome saved to '{filename}'")
    
    pool.close()
    pool.join()


# This makes the script runnable
if __name__ == "__main__":
    print("Starting Parallel GA...")
    print(f"Make sure your {NUM_WORKERS} Docker workers are running on ports {BASE_PORT + 1}-{BASE_PORT + NUM_WORKERS}.")
    run_ga()
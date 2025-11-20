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

# --- CONFIGURATION ---
SEGMENT_STEPS = 100       # Max steps (only used if target not hit)
POPULATION_SIZE = 60      
NUM_GENERATIONS = 20
NUM_ELITES = 4
MUTATION_RATE = 0.15      
DELTA_T = 1/40.0          # Matches car.py fixed time step

NAS_HOME = "/home/jeremy.duc/nas_home/RallyRobotPilot"
RESULTS_DIR = os.path.join(NAS_HOME, "GA/results_segmented")
SEED_FILE = os.path.join(NAS_HOME, "GA/seed/human_seed.json")

POSSIBLE_ACTIONS = [
    'push forward;',
    'push forward; push right;',
    'push forward; push left;',
    'push right;',
    'push left;',
    'push back;',
    'push back; push right;',
    'push back; push left;',
    ''
]

# --- ASSET LOADING ---
def load_checkpoints():
    path = "/home/jeremy.duc/nas_home/RallyRobotPilot/GA/assets/SimpleTrack_checkpoints.json"
    try:
        with open(path, 'r') as f:
            data = json.load(f)['checkpoints']
            
            # --- FIX: SKIP CHECKPOINT 1 ---
            # As requested, we remove Index 1 to fix start line targeting
            if len(data) > 1:
                # print("⚠️ Removing Checkpoint Index 1 from list as requested.")
                del data[1] 
                
            return data
    except:
        return []

CHECKPOINTS = load_checkpoints()

def load_human_seed():
    try:
        if os.path.exists(SEED_FILE):
            print(f"🧬 Loading Human Seed from: {SEED_FILE}")
            with open(SEED_FILE, "r") as f:
                full_seed = json.load(f)
                if len(full_seed) >= SEGMENT_STEPS:
                    return full_seed[:SEGMENT_STEPS]
                else:
                    return full_seed + [''] * (SEGMENT_STEPS - len(full_seed))
        return None
    except: return None

def get_target_coords(cp_data):
    raw = cp_data['position']
    return -raw[0], -raw[1]

# --- DOCKER ---
def run_docker_cmd(rank, cmd_list):
    try:
        subprocess.run(["sg", "docker", "-c", " ".join(cmd_list)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except: return False

def start_docker_container(rank):
    port = 5000 + rank
    cname = f"worker_{rank}_seg"
    run_docker_cmd(rank, ["docker", "rm", "-f", cname])
    args = ["docker", "run", "-d", "--rm", "--name", cname,
            "-v", f"{NAS_HOME}/rallyrobopilot:/app/rallyrobopilot",
            "-v", f"{NAS_HOME}/assets:/app/assets",
            "-p", f"{port}:5000", "rallyrobopilot-headless"]
    if run_docker_cmd(rank, args):
        time.sleep(5)
        return f"http://localhost:{port}", cname
    return None, None

def stop_docker_container(cname):
    if cname: run_docker_cmd(0, ["docker", "kill", cname])

# --- EVALUATION ---
def evaluate_segment(history_actions, new_actions, start_cp_index, target_cp_index, base_url):
    full_genome = history_actions + new_actions
    final_data = None
    steps_taken = 0
    
    try:
        requests.post(f"{base_url}/command", json={'command': 'reset;'}, timeout=1)
        
        for i, action in enumerate(full_genome):
            # 1. Send Action
            requests.post(f"{base_url}/command", json={'command': action}, timeout=0.5)
            
            # 2. Wait for Physics
            time.sleep(DELTA_T)
            steps_taken = i

            # 3. CHECK SENSING EVERY SINGLE FRAME
            # Removed the 'i % 3' check to catch high-speed hits immediately
            res = requests.get(f"{base_url}/sensing", timeout=1).json()
            final_data = res
            
            # 4. IMMEDIATE STOP LOGIC
            if res['lap_progress'] >= target_cp_index:
                # Hit detected! Stop immediately to preserve velocity reading.
                break 

        if final_data is None:
            final_data = requests.get(f"{base_url}/sensing", timeout=1).json()

        progress = final_data.get('lap_progress', 0)
        velocity = final_data.get('car_speed', 0.0)
        
        # --- FITNESS CALCULATION ---
        fitness = 0.0
        
        # 1. Critical Fail (Went backwards or reset)
        if progress < start_cp_index:
            return -5000.0, progress, velocity

        # 2. Wall Penalty (Stuck)
        if abs(velocity) < 1.0:
            fitness -= 2000.0

        # 3. Target Logic
        target_data = CHECKPOINTS[target_cp_index]
        tx, tz = get_target_coords(target_data)
        dist = math.dist((final_data['car_position x'], final_data['car_position z']), (tx, tz))
        
        if progress >= target_cp_index:
            # SUCCESS
            fitness += 10000
            fitness += (velocity * 150.0) 
            
            # EFFICIENCY BONUS: Reward for finishing in fewer frames
            fitness += ((len(full_genome) - steps_taken) * 10.0)
            
            # GRADIENT PENALTY
            # Soft penalty if velocity < 15.0 to encourage speed
            if velocity < 15.0:
                deficit = 15.0 - max(0.0, velocity)
                penalty = deficit * (2000.0 / 15.0) 
                fitness -= penalty
        else:
            # IN PROGRESS
            fitness += (3000.0 / (dist + 1.0))
            fitness += (velocity * 5.0)

        return fitness, progress, velocity

    except:
        return -99999.0, 0, 0.0

def record_trajectory(full_genome, base_url):
    traj = []
    try:
        requests.post(f"{base_url}/command", json={'command': 'reset;'}, timeout=1)
        for i, action in enumerate(full_genome):
            requests.post(f"{base_url}/command", json={'command': action}, timeout=0.5)
            time.sleep(DELTA_T)
            res = requests.get(f"{base_url}/sensing").json()
            traj.append([res['car_position x'], res['car_position z'], res['car_speed']])
        return traj
    except: return []

# --- GA UTILS ---
def create_random_segment():
    return [random.choice(POSSIBLE_ACTIONS) for _ in range(SEGMENT_STEPS)]

def mutate(genome, rate=None):
    effective_rate = rate if rate is not None else MUTATION_RATE
    return [random.choice(POSSIBLE_ACTIONS) if random.random() < effective_rate else g for g in genome]

def crossover(p1, p2):
    pt = random.randint(1, len(p1) - 1)
    return p1[:pt] + p2[pt:]

def get_last_action(global_actions):
    if not global_actions: return 'push forward;' 
    return global_actions[-1]

# --- MAIN ---
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"👑 Master Segmented GA started. Workers: {size-1}")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        human_seed = load_human_seed()
        GLOBAL_BEST_ACTIONS = []
        
        # Iterate through MODIFIED checkpoints list
        for i in range(0, len(CHECKPOINTS) - 1):
            start_node = i
            target_node = i + 1
            
            print(f"\n🚀 OPTIMIZING SEGMENT {i}: CP {start_node} -> CP {target_node}")
            
            # --- POPULATION INITIALIZATION ---
            population = []
            
            if i == 0 and human_seed:
                print("   -> Injecting Human Seed")
                population.append(human_seed)
                for _ in range((POPULATION_SIZE // 2) - 1): 
                    population.append(mutate(human_seed, rate=0.2))
                for _ in range(POPULATION_SIZE - len(population)): 
                    population.append(create_random_segment())
            
            elif i > 0:
                print("   -> Injecting Inertia Seeds")
                last_action = get_last_action(GLOBAL_BEST_ACTIONS)
                momentum_seed = [last_action] * SEGMENT_STEPS
                population.append(momentum_seed)
                for _ in range(15): 
                    population.append(mutate(momentum_seed, rate=0.3))
                for _ in range(POPULATION_SIZE - len(population)): 
                    population.append(create_random_segment())
            else:
                population = [create_random_segment() for _ in range(POPULATION_SIZE)]

            seg_best_fit = -99999
            seg_best_genome = []
            seg_best_vel = 0.0

            for gen in range(NUM_GENERATIONS):
                # Adaptive Mutation
                progress = gen / NUM_GENERATIONS
                current_mutation_rate = max(0.02, MUTATION_RATE * (1.0 - (progress * 0.5)))
                
                results = []
                active_jobs = {}
                next_idx = 0
                
                for w in range(1, size):
                    if next_idx < len(population):
                        comm.send((GLOBAL_BEST_ACTIONS, population[next_idx], start_node, target_node), dest=w, tag=100)
                        active_jobs[w] = population[next_idx]
                        next_idx += 1
                    else: comm.send(None, dest=w, tag=0)

                while len(results) < len(population):
                    status = MPI.Status()
                    fit, prog, vel = comm.recv(source=MPI.ANY_SOURCE, tag=200, status=status)
                    sender = status.Get_source()
                    genome = active_jobs.pop(sender)
                    results.append((fit, genome, prog, vel))
                    
                    if next_idx < len(population):
                        comm.send((GLOBAL_BEST_ACTIONS, population[next_idx], start_node, target_node), dest=sender, tag=100)
                        active_jobs[sender] = population[next_idx]
                        next_idx += 1

                results.sort(key=lambda x: x[0], reverse=True)
                best_fit, best_g, best_p, best_v = results[0]
                success_count = sum(1 for r in results if r[2] >= target_node)

                print(f"  Gen {gen+1:02d}: Best={best_fit:7.1f} | Vel={best_v:5.1f} | Mut={current_mutation_rate:.2f} | Hit?={'✅' if best_p >= target_node else '❌'} ({success_count}/{POPULATION_SIZE})")

                live_data = {
                    "generation": gen,
                    "fitness": best_fit,
                    "velocity": best_v,
                    "genome": GLOBAL_BEST_ACTIONS + best_g
                }
                with open(f"{RESULTS_DIR}/live_spectator.json", "w") as f:
                    json.dump(live_data, f)

                if best_fit > seg_best_fit:
                    seg_best_fit = best_fit
                    seg_best_genome = best_g
                    seg_best_vel = best_v

                new_pop = [r[1] for r in results[:NUM_ELITES]]
                while len(new_pop) < POPULATION_SIZE:
                    p = random.sample(results[:12], 2) 
                    child = crossover(p[0][1], p[1][1])
                    new_pop.append(mutate(child, rate=current_mutation_rate))
                population = new_pop

            if seg_best_fit > 5000 and seg_best_vel >= 15.0:
                print(f"✅ Locking Segment {i}. Exit Velocity: {seg_best_vel:.1f}")
                GLOBAL_BEST_ACTIONS.extend(seg_best_genome)
                
                # Get trajectory for the full path so far
                comm.send(GLOBAL_BEST_ACTIONS, dest=1, tag=300)
                traj = comm.recv(source=1, tag=300)

                # --- NEW METADATA STRUCTURE ---
                output_data = {
                    "metadata": {
                        "file_format_version": "1.0",
                        "script": os.path.basename(__file__),
                        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
                        "segment_index": i,
                        "start_checkpoint_index": start_node,
                        "target_checkpoint_index": target_node
                    },
                    "ga_config": {
                        "population_size": POPULATION_SIZE,
                        "num_generations_per_segment": NUM_GENERATIONS,
                        "num_elites": NUM_ELITES,
                        "initial_mutation_rate": MUTATION_RATE,
                        "segment_steps": SEGMENT_STEPS
                    },
                    "segment_result": {
                        "best_fitness": seg_best_fit,
                        "exit_velocity": seg_best_vel,
                        "segment_genome": seg_best_genome
                    },
                    "cumulative_result": {
                        "total_actions": GLOBAL_BEST_ACTIONS,
                        "trajectory": traj
                    }
                }
                
                # --- NEW FILENAME ---
                filename = f"genome_segment_{i:02d}.json"
                filepath = os.path.join(RESULTS_DIR, filename)

                with open(filepath, "w") as f:
                    json.dump(output_data, f, indent=4)
                
                print(f"   -> Saved segment results to {filename}")
            else:
                print(f"❌ SEGMENT FAILED. Best Vel: {seg_best_vel:.1f} (Req: 15.0). Stopping.")
                for w in range(1, size): comm.send(None, dest=w, tag=666)
                sys.exit(1)

        for w in range(1, size): comm.send(None, dest=w, tag=666)

    else:
        time.sleep((rank % 10) * 1)
        base_url, cname = start_docker_container(rank)
        try:
            while True:
                status = MPI.Status()
                packet = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
                tag = status.Get_tag()
                
                if tag == 666: break 
                if tag == 0: time.sleep(0.1); continue 
                
                if tag == 100: 
                    res = evaluate_segment(packet[0], packet[1], packet[2], packet[3], base_url) if base_url else (-999,0,0)
                    comm.send(res, dest=0, tag=200)
                    
                if tag == 300: 
                    res = record_trajectory(packet[0], base_url) if base_url else []
                    comm.send(res, dest=0, tag=300)
        except: pass
        finally: stop_docker_container(cname)

if __name__ == "__main__":
    main()
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
SEGMENT_STEPS = 300       # 7.5s horizon (Essential for the longer start segment)
POPULATION_SIZE = 60      # Full cluster power
NUM_GENERATIONS = 20
NUM_ELITES = 4
MUTATION_RATE = 0.10      # High mutation to find the first turn
DELTA_T = 1/40.0

NAS_HOME = "/home/jeremy.duc/nas_home/RallyRobotPilot"
RESULTS_DIR = os.path.join(NAS_HOME, "GA/results_segmented")

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
            return json.load(f)['checkpoints']
    except:
        return []

CHECKPOINTS = load_checkpoints()

def get_target_coords(cp_data):
    raw = cp_data['position']
    return -raw[0], -raw[1]

# --- DOCKER MANAGEMENT ---
def run_docker_cmd(rank, cmd_list):
    try:
        subprocess.run(["sg", "docker", "-c", " ".join(cmd_list)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except:
        return False

def start_docker_container(rank):
    port = 5000 + rank
    cname = f"worker_{rank}_seg"
    run_docker_cmd(rank, ["docker", "rm", "-f", cname])
    if run_docker_cmd(rank, ["docker", "run", "-d", "--rm", "--name", cname, "-p", f"{port}:5000", "rallyrobopilot-headless"]):
        time.sleep(5)
        return f"http://localhost:{port}", cname
    return None, None

def stop_docker_container(cname):
    if cname: run_docker_cmd(0, ["docker", "kill", cname])

# --- EVALUATION LOGIC ---
def evaluate_segment(history_actions, new_actions, start_cp_index, target_cp_index, base_url):
    full_genome = history_actions + new_actions
    hit_frame = -1
    BUFFER = 20
    
    final_data = None
    
    try:
        requests.post(f"{base_url}/command", json={'command': 'reset;'}, timeout=1)
        
        for i, action in enumerate(full_genome):
            requests.post(f"{base_url}/command", json={'command': action}, timeout=0.5)
            time.sleep(DELTA_T)

            # Smart Polling
            if i % 5 == 0 or hit_frame != -1:
                res = requests.get(f"{base_url}/sensing", timeout=1).json()
                final_data = res
                
                # Check strictly for the TARGET index
                if res['lap_progress'] >= target_cp_index and hit_frame == -1:
                    hit_frame = i
                
                # Exit early if we passed it + buffer
                if hit_frame != -1 and (i - hit_frame) >= BUFFER:
                    break

        if final_data is None:
            final_data = requests.get(f"{base_url}/sensing", timeout=1).json()

        progress = final_data.get('lap_progress', 0)
        velocity = final_data.get('car_speed', 0.0)
        
        # --- FITNESS FUNCTION ---
        fitness = 0.0
        
        # 1. Critical Fail: Behind Start
        # (Allowed for Segment 0/1 consolidation, strict for others)
        if start_cp_index > 0 and progress < start_cp_index:
            return -5000.0, progress, velocity

        # 2. Target Logic
        target_data = CHECKPOINTS[target_cp_index]
        tx, tz = get_target_coords(target_data)
        dist = math.dist((final_data['car_position x'], final_data['car_position z']), (tx, tz))
        
        if progress >= target_cp_index:
            # SUCCESS
            fitness += 10000
            fitness += (velocity * 50.0) # Reward speed
            fitness += ((len(full_genome) - i) * 5.0) # Reward time saved
            
            # Penalty only applied inside success to differentiate "Good Hit" vs "Bad Hit"
            if velocity < 15.0:
                fitness -= 2000.0 # Soft penalty, still better than missing
        else:
            # APPROACHING
            fitness += (3000.0 / (dist + 1.0))
            # Small speed bonus to prevent stopping
            fitness += (velocity * 2.0)

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

def mutate(genome):
    return [random.choice(POSSIBLE_ACTIONS) if random.random() < MUTATION_RATE else g for g in genome]

def crossover(p1, p2):
    pt = random.randint(1, len(p1) - 1)
    return p1[:pt] + p2[pt:]

# --- MAIN ---
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"👑 Master Segmented GA started. Workers: {size-1}")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        GLOBAL_BEST_ACTIONS = []
        
        # --- COMBINED START STRATEGY ---
        # We skip the trivial Start->CP1 segment.
        # We start directly by optimizing Start->CP2.
        # Start Index: 0 (Actual Start)
        # Target Index: 2 (Checkpoint 2)
        
        print("\n⏩ CONSOLIDATING START: Targeting CP 2 directly.")
        
        # Manually set the first target to 2
        # Then standard loop will pick up from there
        
        # Loop targets: 2, 3, 4... up to end
        # Start nodes: 0, 2, 3, 4...
        
        targets = list(range(2, len(CHECKPOINTS)))
        starts = [0] + list(range(2, len(CHECKPOINTS)-1))
        
        for start_node, target_node in zip(starts, targets):
            target_id = CHECKPOINTS[target_node]['id']
            print(f"\n🚀 OPTIMIZING SEGMENT: CP {start_node} -> CP {target_node} (Target ID {target_id})")
            
            population = [create_random_segment() for _ in range(POPULATION_SIZE)]
            seg_best_fit = -99999
            seg_best_genome = []
            seg_best_vel = 0.0

            for gen in range(NUM_GENERATIONS):
                results = []
                active_jobs = {}
                next_idx = 0
                
                # Distribute
                for w in range(1, size):
                    if next_idx < len(population):
                        comm.send((GLOBAL_BEST_ACTIONS, population[next_idx], start_node, target_node), dest=w, tag=100)
                        active_jobs[w] = population[next_idx]
                        next_idx += 1
                    else: comm.send(None, dest=w, tag=0)

                # Collect
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
                # --- NEW: EXPORT FOR LIVE VIEWING ---
                live_data = {
                    "generation": gen,
                    "fitness": best_fit,
                    "velocity": best_v,
                    "genome": GLOBAL_BEST_ACTIONS + best_g # Combine History + New
                }
                with open(f"{RESULTS_DIR}/live_spectator.json", "w") as f:
                    json.dump(live_data, f)
                # ------------------------------------
                
                success_count = sum(1 for r in results if r[2] >= target_node)

                print(f"  Gen {gen+1:02d}: Best={best_fit:7.1f} | Vel={best_v:5.1f} | Hit?={'✅' if best_p >= target_node else '❌'} ({success_count}/{POPULATION_SIZE})")

                if best_fit > seg_best_fit:
                    seg_best_fit = best_fit
                    seg_best_genome = best_g
                    seg_best_vel = best_v

                new_pop = [r[1] for r in results[:NUM_ELITES]]
                while len(new_pop) < POPULATION_SIZE:
                    p = random.sample(results[:10], 2)
                    new_pop.append(mutate(crossover(p[0][1], p[1][1])))
                population = new_pop

            # Lock Logic
            if seg_best_fit > 5000:
                print(f"✅ Locking Segment (Target {target_node}).")
                GLOBAL_BEST_ACTIONS.extend(seg_best_genome)
                
                # Save Viz
                comm.send(GLOBAL_BEST_ACTIONS, dest=1, tag=300)
                traj = comm.recv(source=1, tag=300)
                with open(f"{RESULTS_DIR}/viz_seg_{target_node}.json", "w") as f:
                    json.dump({"actions": GLOBAL_BEST_ACTIONS, "trajectory": traj}, f)
            else:
                print(f"❌ SEGMENT FAILED. Stopping.")
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
                    res = record_trajectory(packet, base_url) if base_url else []
                    comm.send(res, dest=0, tag=300)
        except: pass
        finally: stop_docker_container(cname)

if __name__ == "__main__":
    main()
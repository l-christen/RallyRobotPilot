import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. Data Extraction from Logs ---
# Data manually parsed from the user-provided logs.

# Run 1: Multi-segmented Run
gens = np.arange(1, 21)

# Data for Segment 0 (Success)
fitness_segment_0 = [
    16156.2, 15312.5, 13437.5, 10947.9, 10062.5, 9531.2, 9531.2, 10770.8,
    10416.7, 9531.2, 9531.2, 7177.1, 9354.2, 9531.2, 9531.2, 9354.2,
    9531.2, 7157.1, 7000.0, 6906.2
]

# Data for Segment 1 (Failure)
fitness_segment_1 = [-1988.1] * 20

# Run 2: One-lap-segment Run (Technical Failure & Stagnation)
fitness_one_lap = [
    93.76, 93.79, 93.79, 93.79, 93.79, 93.79, 93.79, 93.79, 93.79, 93.79,
    93.79, 93.79, 93.79, 93.79, 93.79, 93.79, 93.79, 93.79, 93.79, 93.79
]

# --- 2. Create the Comparison Plot ---
print("🎨 Generating comparison plot...")
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 8))

# Plot data for each run
plt.plot(gens, fitness_segment_0, 'o-', label='Multi-Segment: Segment 0 (Success)', color='royalblue', linewidth=2)
plt.plot(gens, fitness_segment_1, 's--', label='Multi-Segment: Segment 1 (Failure)', color='firebrick', linewidth=2)
plt.plot(gens, fitness_one_lap, '^-', label='One-Lap Run (Stagnation & Technical Failure)', color='forestgreen', linewidth=1, markersize=4)

# --- 3. Formatting and Labels ---
plt.title('GA Performance Comparison: Segmented vs. One-Lap Approach', fontsize=16, pad=20)
plt.xlabel('Generation', fontsize=12)
plt.ylabel('Best Fitness Score', fontsize=12)
plt.xticks(np.arange(1, 21, 1)) # Ensure integer ticks for generations
plt.yscale('symlog') # Use a symmetric logarithmic scale to handle large positive and negative values
plt.grid(True, which="both", ls="--")
plt.legend(fontsize=10)



# --- 4. Save the Plot ---
analysis_dir = os.path.join(os.path.dirname(__file__))
if not os.path.exists(analysis_dir):
    os.makedirs(analysis_dir)
    
plot_path = os.path.join(analysis_dir, 'ga_run_comparison.png')
plt.savefig(plot_path)

print(f"✅ Comparison plot saved to: {plot_path}")

#!/bin/bash
#SBATCH --job-name=rally_ga
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --partition=Calypso

# 1. Nettoyage préventif
./manage_workers.sh stop 32

# --- AJOUT CRITIQUE ICI ---
echo "🏗️ Construction de l'image Docker sur ce noeud..."
docker-compose build headless
# --------------------------

# 2. Démarrage des workers
./manage_workers.sh start 32

# 3. Attente démarrage Flask
sleep 10

# 4. Lancement GA
python3 -u scripts/run_ga.py

# 5. Nettoyage final
./manage_workers.sh stop 32
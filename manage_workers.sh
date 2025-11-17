#!/bin/bash

# --- Config ---
# 1. Build your image first: docker-compose build headless
# 2. Find its name: docker images | grep headless
# 3. Paste that name here:
DOCKER_IMAGE_NAME="rallyrobotpilot_headless"
# --- End Config ---

# Get the command (start|stop) and number of workers
COMMAND=$1
NUM_WORKERS=$2

# --- Base port to start from ---
BASE_PORT=5000

if [ -z "$COMMAND" ] || [ -z "$NUM_WORKERS" ]; then
  echo "Usage: $0 <start|stop> <number_of_workers>"
  echo "Example: $0 start 32"
  exit 1
fi

if [ "$COMMAND" == "start" ]; then
  echo "Starting $NUM_WORKERS workers..."
  for i in $(seq 1 $NUM_WORKERS); do
    PORT=$((BASE_PORT + i))
    WORKER_NAME="worker_${PORT}"
    
    echo "  Starting $WORKER_NAME on port $PORT"
    docker run -d --rm -p ${PORT}:5000 --name ${WORKER_NAME} ${DOCKER_IMAGE_NAME}
  done
  echo "All workers started."

elif [ "$COMMAND" == "stop" ]; then
  echo "Stopping $NUM_WORKERS workers..."
  for i in $(seq 1 $NUM_WORKERS); do
    PORT=$((BASE_PORT + i))
    WORKER_NAME="worker_${PORT}"
    
    echo "  Stopping $WORKER_NAME"
    docker stop ${WORKER_NAME}
  done
  echo "All workers stopped and removed."

else
  echo "Invalid command: $COMMAND. Use 'start' or 'stop'."
  exit 1
fi
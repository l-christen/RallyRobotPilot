![](https://gitlab.hevs.ch/uploads/-/system/project/avatar/1282/rallyrobopilot.jpg)

# Data computing & learning project - learning virtual rally cars visual guiding and optimal piloting with a distributed framework


## Project Overview

This repository contains the sandbox driving simulation for the 5th-semester BSc "Data computing & learning project". The objective is to develop a comprehensive, data-driven solution to pilot a virtual rally car using **computer vision exclusively**.

This project is held at the **Haute Ecole d'Ingénierie de Sion** in Switzerland, under the supervision of Prof. Dr. Louis Lettry, Dr. Florian Desmon and Dr. Cédric Travelletti.

---

### Core Project Objectives

1.  **Vision-Based Piloting**
    The control system must be based **solely on image data**. At evaluation time, the model will not have access to any explicit state information such as raycasts, speed, location, or rotation data.

2.  **Optimal Piloting**
    The goal extends beyond simple navigation; the agent must demonstrate **optimal piloting**. This requires defining and implementing a strategy to learn and follow optimal racing lines, similar to those shown in theory (e.g., classic racing line, hairpin bends, double apex), to achieve the best possible performance.

3.  **Distributed Framework**
    The entire solution must be engineered as an **automated and scalable process** designed to operate on a distributed cluster. This includes automated model performance monitoring and will leverage technologies from Module 302 (MPI).

4.  **Data-Driven Approach**
    A machine learning model (e.g., SVM, Random Forest, Neural Network) is required. A significant portion of the project involves a thorough analysis of data collection procedures, training data (bias, augmentation), and training analysis.
---

### New Challenges (Novelties)

This project introduces new track features that the autonomous agent must successfully navigate:
* **Track Intersections**
* **Direction Indicators**
* **Moving Obstacles**
---
## Installation Instructions
This project use Python 3.13.5. To install the required dependencies, it is recommended to use a virtual environment.

Create and activate a virtual environment:
```
python -m venv venv
```
On windows:
```
venv\Scripts\activate
```
On Unix or MacOS:
```
source venv/bin/activate
```

Then, install the required dependencies using pip:
```
pip install -r requirements.txt
```

Also install the package in editable mode:
```
pip install -e .
```

Finally, to test and run the game, you can use
```
python scripts/main.py
```
---

## About the RallyRoboPilot Simulation
Launching main.py starts a race with a single car on the provided track. 
This track can be controlled either by keyboard (*AWSD*) or by a socket interface. 

### Car Sensing
The car sensing is available in two commodities: **raycasts** and **images**. These sensing snapshots are sent at 10 Hertz (i.e. 10 times a second). Due to this fact, correct reception of snapshot messages has to be done regularly (See Server buffer saturation section).

### Communication Protocol

A remote controller can be implemented using TCP socket connecting on localhost on port 7654. 
Different commands can be issued to the race simulation to control the car.


### Control Commands
The control commands are the following:

| Command      | Description                                     |
|--------------|-------------------------------------------------|
| w            | Accelerate the car                              |
| s            | Brake / Reverse the car                         |
| a            | Steer left                                      |
| d            | Steer right                                     |
| g            | Respawn the car                                 |
| Esc          | Quit the simulation                             |

---
### Docker Containers
Provided by the professors, two docker containers are available: one that launches the usual graphic version (only runs on Linux host with X11), and one that runs the game on a headless server (will be used for model training).


The graphical container can be run using


`docker compose run gui`


and the headless one can be run using 


`docker compose run headless`


# Credits
The game code is based on the repository [https://github.com/mandaw2014/Rally](https://github.com/mandaw2014/Rally)

# CPU Scheduler Simulation

A comprehensive simulation and visualization tool for CPU scheduling algorithms used in operating systems.

## Overview

This application simulates different CPU scheduling algorithms and provides visual comparisons of their performance. It supports multiple scheduling algorithms, process generation methods, and comprehensive performance metrics to help understand the behavior and efficiency of each algorithm.

## Features

- **Multiple Scheduling Algorithms**:
  - First-Come, First-Served (FCFS)
  - Shortest Job First (SJF)
  - Priority Scheduling
  - Round Robin (RR)
  - Priority Round Robin

- **Process Management**:
  - Random process generation with customizable parameters
  - Manual process creation
  - Import/export processes from/to JSON files

- **Visualization**:
  - Gantt charts for CPU execution timeline
  - Process timelines showing states over time
  - Performance metrics comparisons
  - Algorithm comparison across multiple metrics

- **Metrics**:
  - Average Turnaround Time
  - Average Waiting Time  
  - Average Response Time
  - CPU Utilization
  - Throughput
  - Algorithm-specific metrics (context switches, fairness, etc.)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ykram051/CPU-Scheduler-Simulation.git
cd CPU-Scheduler-Simulation
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Run the simulation from the command line:
```bash
python main.py
```

### Interactive Web Interface

For a more interactive experience with visualizations:
```bash
python -m streamlit run app.py
```

This will open a web interface in your browser where you can:
1. Generate or import processes
2. Select a scheduling algorithm
3. Run simulations and visualize results
4. Compare algorithm performance

## Process Format

When importing processes from a JSON file, use the following format:
```json
{
    "processes": [
        {
            "pid": 1,
            "arrival_time": 0,
            "burst_time": 5,
            "priority": 2
        },
        {
            "pid": 2,
            "arrival_time": 1,
            "burst_time": 3,
            "priority": 1
        }
    ]
}
```

## Project Structure

- `app.py` - Streamlit web application for interactive visualization
- `main.py` - Command-line interface for running simulations
- `metrics.py` - Performance metrics calculation functions
- `process.py` - Process class definition
- `InputHandler.py` - Process generation and I/O utilities
- `visualization.py` - Visualization functions for command-line outputs
- `SchedulingAlgorithms/` - Directory containing all scheduling algorithm implementations:
  - `Scheduler.py` - Base class for all schedulers
  - `FirstComeFirstServe.py` - FCFS algorithm implementation
  - `ShortestJobFirst.py` - SJF algorithm implementation
  - `PriorityScheduling.py` - Priority scheduling implementation
  - `RoundRobin.py` - Round Robin scheduling implementation
  - `PriorityRoundRobin.py` - Priority-based Round Robin implementation

## Design Decisions

1. **Object-Oriented Approach**: Each scheduling algorithm inherits from a base `Scheduler` class, allowing for consistent interfaces and clear organization.

2. **Separation of Concerns**: The project separates process management, algorithm implementation, metrics calculation, and visualization into distinct modules.

3. **Algorithm Comparability**: All scheduling algorithms return standardized metrics, enabling direct comparison between different approaches.

4. **Interactive Visualization**: The Streamlit web interface provides intuitive visualization options to help understand algorithm behavior.

5. **Extensibility**: New scheduling algorithms can be easily added by creating a new class that inherits from the base `Scheduler` class.


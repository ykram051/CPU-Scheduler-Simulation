# CPU Scheduler Simulation

A simulation tool for CPU scheduling algorithms used in operating systems.

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

## Running the Program

### Web Interface (Recommended)

For an interactive experience with visualizations:
```bash
python -m streamlit run app.py
```

This will open a web interface in your browser where you can:
1. Generate or define processes
2. Select scheduling algorithms
3. Run simulations and visualize results
4. Compare algorithm performance

#### Using Pre-defined Process Files

The application supports importing process data from JSON files:

1. You can use the provided `temp_processes.json` file as a starting point
2. Upload it through the "Load from file" option in the web interface
3. Create your own process files following the same format

Example process file format:
```json
[
  {"pid": 1, "arrival_time": 0, "burst_time": 5, "priority": 3},
  {"pid": 2, "arrival_time": 2, "burst_time": 3, "priority": 1},
  {"pid": 3, "arrival_time": 4, "burst_time": 6, "priority": 2}
]
```

### Command Line

To run the simulation from the command line:
```bash
python main.py [options]
```

Example commands:
```bash
# Run all algorithms with default processes
python main.py --all

# Generate 10 random processes and run FCFS and SJF
python main.py --random 10 --fcfs --sjf

# Load processes from a file and save visualizations
python main.py --file my_processes.json --all --output results
```

## Implementation Details

This application implements six CPU scheduling algorithms:
- First-Come, First-Served (FCFS)
- Shortest Job First (SJF)
- Priority Scheduling
- Round Robin (RR)
- Priority Round Robin
- Multilevel Feedback Queue (MFQ)

Each algorithm calculates key performance metrics including average waiting time, turnaround time, response time, CPU utilization, and algorithm-specific metrics.

The visualization components generate Gantt charts, process lifecycle visualizations, and comparative metrics to help understand algorithm behavior.

## Project Structure

- `app.py` - Streamlit web application
- `main.py` - Command-line interface
- `SchedulingAlgorithms/` - Individual algorithm implementations
- `visualization.py` & `advanced_visualizations.py` - Visualization functions
- `process.py` - Process class definition
- `metrics.py` - Performance metrics calculation

## Testing

The project includes a test suite to verify the correctness of the scheduling algorithms:

```bash
# Run the test suite
python test_scheduler_simulation.py
```

The test scheduler validates:
- Algorithm implementations
- Metrics calculations
- Edge case handling
- Process state transitions

This ensures that all scheduling algorithms produce correct results and meet the expected performance characteristics.



import os
import argparse
import csv
from process import Process
import input_handler
import metrics
import visualization
from SchedulingAlgorithms.Scheduler import Scheduler
from SchedulingAlgorithms.FirstComeFirstServe import FCFSScheduler
from SchedulingAlgorithms.ShortestJobFirst import SJFScheduler
from SchedulingAlgorithms.PriorityScheduling import PriorityScheduler
from SchedulingAlgorithms.RoundRobin import RoundRobinScheduler
from SchedulingAlgorithms.PriorityRoundRobin import PriorityRRScheduler

def parse_arguments():
    """Parse command line arguments for the CPU scheduler simulation"""
    parser = argparse.ArgumentParser(description="CPU Scheduler Simulation")
    
    # Input options
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument("--random", type=int, help="Generate N random processes")
    input_group.add_argument("--file", type=str, help="Read processes from CSV file")
    input_group.add_argument("--save_input", type=str, help="Save generated processes to CSV file")
    
    # Process generation parameters
    gen_group = parser.add_argument_group('Process Generation Parameters')
    gen_group.add_argument("--arrival_min", type=int, default=0, help="Minimum arrival time (default: 0)")
    gen_group.add_argument("--arrival_max", type=int, default=20, help="Maximum arrival time (default: 20)")
    gen_group.add_argument("--burst_min", type=int, default=1, help="Minimum burst time (default: 1)")
    gen_group.add_argument("--burst_max", type=int, default=20, help="Maximum burst time (default: 20)")
    gen_group.add_argument("--priority_min", type=int, default=1, help="Minimum priority (default: 1)")
    gen_group.add_argument("--priority_max", type=int, default=10, help="Maximum priority (default: 10)")
    
    # Scheduling algorithm options
    alg_group = parser.add_argument_group('Scheduling Algorithms')
    alg_group.add_argument("--fcfs", action="store_true", help="First-Come-First-Served scheduling")
    alg_group.add_argument("--sjf", action="store_true", help="Shortest Job First scheduling")
    alg_group.add_argument("--priority", action="store_true", help="Priority scheduling")
    alg_group.add_argument("--rr", action="store_true", help="Round Robin scheduling")
    alg_group.add_argument("--priority_rr", action="store_true", help="Priority + Round Robin scheduling")
    alg_group.add_argument("--all", action="store_true", help="Run all scheduling algorithms")
    alg_group.add_argument("--time_quantum", type=int, default=4, help="Time quantum for RR and Priority RR (default: 4)")
    
    # Visualization options
    vis_group = parser.add_argument_group('Visualization Options')
    vis_group.add_argument("--gantt", action="store_true", help="Generate Gantt charts")
    vis_group.add_argument("--timeline", action="store_true", help="Generate process timelines")
    vis_group.add_argument("--compare", action="store_true", help="Generate comparison charts")
    vis_group.add_argument("--output_dir", type=str, help="Directory to save visualizations")
    vis_group.add_argument("--show", action="store_true", help="Show charts instead of saving")
    
    args = parser.parse_args()
    
    # If no algorithm is specified, default to running all
    if not any([args.fcfs, args.sjf, args.priority, args.rr, args.priority_rr]):
        args.all = True
        
    return args

def generate_processes(args):
    """Generate or load processes based on command line arguments"""
    processes = []
    
    if args.random:
        # Generate random processes
        processes = input_handler.generate_random_processes(
            args.random,
            arrival_min=args.arrival_min, 
            arrival_max=args.arrival_max,
            burst_min=args.burst_min, 
            burst_max=args.burst_max,
            priority_min=args.priority_min, 
            priority_max=args.priority_max
        )
        print(f"Generated {len(processes)} random processes")
        
    elif args.file:
        # Load processes from file
        processes = input_handler.read_processes_from_file(args.file)
        print(f"Loaded {len(processes)} processes from {args.file}")
    else:
        # Default to 5 random processes if no input method specified
        processes = input_handler.generate_random_processes(5)
        print(f"Generated {len(processes)} default random processes")
    
    # Save processes to file if requested
    if args.save_input:
        input_handler.save_processes_to_file(processes, args.save_input)
    
    return processes

def run_simulations(processes, args):
    """Run all the selected scheduling algorithms on the processes"""
    results = {}
    time_quantum = args.time_quantum
    
    # Reset all processes before each simulation
    def reset_processes():
        for process in processes:
            process.reset()
    
    # First-Come-First-Served
    if args.fcfs or args.all:
        reset_processes()
        print("\nRunning FCFS scheduling...")
        fcfs = FCFSScheduler(processes)
        fcfs.schedule()
        fcfs_metrics = metrics.calculate_fcfs_metrics(fcfs.completed_processes, fcfs.execution_sequence)
        
        results["FCFS"] = {
            'execution_sequence': fcfs.execution_sequence,
            'completed_processes': fcfs.completed_processes,
            **fcfs_metrics
        }
        
        print_metrics("FCFS", fcfs_metrics)
    
    # Shortest Job First
    if args.sjf or args.all:
        reset_processes()
        print("\nRunning SJF scheduling...")
        sjf = SJFScheduler(processes)
        sjf.schedule()
        sjf_metrics = metrics.calculate_sjf_metrics(sjf.completed_processes, sjf.execution_sequence)
        
        results["SJF"] = {
            'execution_sequence': sjf.execution_sequence,
            'completed_processes': sjf.completed_processes,
            **sjf_metrics
        }
        
        print_metrics("SJF", sjf_metrics)
    
    # Priority Scheduling
    if args.priority or args.all:
        reset_processes()
        print("\nRunning Priority scheduling...")
        priority = PriorityScheduler(processes)
        priority.schedule()
        priority_metrics = metrics.calculate_priority_metrics(priority.completed_processes, priority.execution_sequence)
        
        results["Priority"] = {
            'execution_sequence': priority.execution_sequence,
            'completed_processes': priority.completed_processes,
            **priority_metrics
        }
        
        print_metrics("Priority", priority_metrics)
    
    # Round Robin
    if args.rr or args.all:
        reset_processes()
        print(f"\nRunning Round Robin scheduling (Time Quantum: {time_quantum})...")
        rr = RoundRobinScheduler(processes, time_quantum)
        rr.schedule()
        rr_metrics = metrics.calculate_rr_metrics(rr.completed_processes, rr.execution_sequence, time_quantum)
        
        results["Round Robin"] = {
            'execution_sequence': rr.execution_sequence,
            'completed_processes': rr.completed_processes,
            **rr_metrics
        }
        
        print_metrics("Round Robin", rr_metrics)
    
    # Priority + Round Robin
    if args.priority_rr or args.all:
        reset_processes()
        print(f"\nRunning Priority + Round Robin scheduling (Time Quantum: {time_quantum})...")
        priority_rr = PriorityRRScheduler(processes, time_quantum)
        priority_rr.schedule()
        priority_rr_metrics = metrics.calculate_priority_rr_metrics(
            priority_rr.completed_processes, 
            priority_rr.execution_sequence,
            time_quantum
        )
        
        results["Priority RR"] = {
            'execution_sequence': priority_rr.execution_sequence,
            'completed_processes': priority_rr.completed_processes,
            **priority_rr_metrics
        }
        
        print_metrics("Priority RR", priority_rr_metrics)
    
    # Compare algorithms if multiple were run
    if len(results) > 1:
        print("\nComparison of algorithms:")
        # Extract metrics for comparison (excluding sequences and processes)
        comparison_data = {}
        for alg, result in results.items():
            comparison_data[alg] = {
                k: v for k, v in result.items() 
                if k not in ['execution_sequence', 'completed_processes'] and not isinstance(v, dict)
            }
        
        comparison = metrics.compare_algorithms(comparison_data)
        print_comparison(comparison)
    
    return results

def print_metrics(algorithm_name, metrics_data):
    """Print formatted metrics for an algorithm"""
    print(f"\n{algorithm_name} Metrics:")
    print("-" * 50)
    
    # Print basic metrics
    basic_metrics = [
        ('Average Turnaround Time', 'avg_turnaround_time'),
        ('Average Waiting Time', 'avg_waiting_time'),
        ('Average Response Time', 'avg_response_time'),
        ('Throughput (processes/time unit)', 'throughput'),
        ('CPU Utilization (%)', 'cpu_utilization')
    ]
    
    for name, key in basic_metrics:
        if key in metrics_data:
            value = metrics_data[key]
            # Format CPU utilization as percentage
            if key == 'cpu_utilization':
                print(f"{name}: {value:.2f}%")
            else:
                print(f"{name}: {value:.2f}")
    
    # Print algorithm-specific metrics
    if 'context_switches' in metrics_data:
        print(f"Context Switches: {metrics_data['context_switches']}")
    
    if 'fairness' in metrics_data:
        print(f"Process Fairness: {metrics_data['fairness']:.2f}")
    
    if 'quantum_utilization' in metrics_data:
        print(f"Time Quantum Utilization: {metrics_data['quantum_utilization']:.2f}")
    
    if 'waiting_time_by_priority' in metrics_data:
        print("\nWaiting Time by Priority:")
        priorities = sorted(metrics_data['waiting_time_by_priority'].keys())
        for priority in priorities:
            wt = metrics_data['waiting_time_by_priority'][priority]
            print(f"  Priority {priority}: {wt:.2f}")
    
    print("-" * 50)

def print_comparison(comparison):
    """Print comparison results between algorithms"""
    print("\nAlgorithm Comparison:")
    print("=" * 60)
    
    for metric, data in comparison.items():
        best_alg = data['best_algorithm']
        values = data['values']
        
        # Format metric name for display
        display_name = metric.replace('_', ' ').title()
        print(f"\n{display_name}:")
        print("-" * 40)
        
        # Print values for each algorithm
        for alg, value in values.items():
            # Add marker for best algorithm
            marker = "★" if alg == best_alg else " "
            
            # Format CPU utilization as percentage
            if metric == 'cpu_utilization':
                print(f"{marker} {alg}: {value:.2f}%")
            else:
                print(f"{marker} {alg}: {value:.2f}")
    
    print("\n" + "=" * 60)

def main():
    """Main function to run the CPU scheduler simulation"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Generate or load processes
    processes = generate_processes(args)
    
    # Print process details
    print("\nProcess Details:")
    for process in sorted(processes, key=lambda p: p.pid):
        print(f"  {process}")
    
    # Run simulations
    results = run_simulations(processes, args)
    
    # Generate visualizations if requested
    if args.gantt or args.timeline or args.compare:
        print("\nGenerating visualizations...")
        
        # Determine where to save or show visualizations
        save_dir = None if args.show else args.output_dir
        
        # If saving but no directory specified, use default
        if save_dir is None and not args.show:
            save_dir = 'cpu_scheduler_results'
            print(f"No output directory specified, using '{save_dir}'")
            
            # Create directory if it doesn't exist
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        
        # Generate comprehensive visualizations
        visualization.create_comprehensive_report(results, save_dir)
        
        if args.show:
            print("Visualizations displayed. Close the windows to continue.")
        else:
            print(f"Visualizations saved to {save_dir}")
    
    print("\nSimulation complete.")

if __name__ == "__main__":
    main()
import argparse
import json
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from process import Process
from SchedulingAlgorithms.FirstComeFirstServe import FCFSScheduler
from SchedulingAlgorithms.ShortestJobFirst import SJFScheduler
from SchedulingAlgorithms.PriorityScheduling import PriorityScheduler
from SchedulingAlgorithms.RoundRobin import RoundRobinScheduler
from SchedulingAlgorithms.PriorityRoundRobin import PriorityRRScheduler
from SchedulingAlgorithms.MultilevelFeedbackQueue import MFQScheduler

import visualization
import advanced_visualizations
from config import get_config

def generate_random_processes(count):
    """Generate random processes for simulation"""
    random_config = get_config("random_process")
    processes = []
    
    for i in range(1, count + 1):
        arrival_time = random.randint(
            random_config["min_arrival_time"], 
            random_config["max_arrival_time"]
        )
        burst_time = random.randint(
            random_config["min_burst_time"], 
            random_config["max_burst_time"]
        )
        priority = random.randint(
            random_config["min_priority"], 
            random_config["max_priority"]
        )
        processes.append(Process(i, arrival_time, burst_time, priority))
        
    return processes

def save_processes_to_file(processes, filename):
    """Save processes to a JSON file"""
    with open(filename, 'w') as f:
        json.dump([{
            'pid': p.pid,
            'arrival_time': p.arrival_time,
            'burst_time': p.burst_time,
            'priority': p.priority
        } for p in processes], f, indent=4)

def load_processes_from_file(filename):
    """Load processes from a JSON file"""
    with open(filename, 'r') as f:
        data = json.load(f)
        return [Process(
            p['pid'], 
            p['arrival_time'], 
            p['burst_time'], 
            p['priority']
        ) for p in data]

def run_simulation(processes, algorithms, output_dir=None):
    """Run simulation with specified algorithms and processes"""
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = {}
    
    for name, algorithm in algorithms.items():
        # Reset all processes for each algorithm
        for p in processes:
            p.reset()
            
        # Run the algorithm
        result = algorithm(processes.copy()).schedule()
        results[name] = result
        
        print(f"\n{name} Results:")
        print(f"Average waiting time: {result['avg_waiting_time']:.2f}")
        print(f"Average turnaround time: {result['avg_turnaround_time']:.2f}")
        print(f"Average response time: {result['avg_response_time']:.2f}")
        if 'cpu_utilization' in result:
            print(f"CPU utilization: {result['cpu_utilization']:.2f}%")
            
        # Additional algorithm-specific metrics
        if 'context_switches' in result:
            print(f"Context switches: {result['context_switches']}")
        if 'fairness' in result:
            print(f"Fairness index: {result['fairness']:.2f}")
        if 'avg_queue_level' in result:
            print(f"Average queue level: {result['avg_queue_level']:.2f}")
        if 'top_queue_ratio' in result:
            print(f"Top queue ratio: {result['top_queue_ratio']:.2f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="CPU Scheduler Simulation")
    parser.add_argument('--random', type=int, help='Generate random processes')
    parser.add_argument('--file', type=str, help='Load processes from file')
    parser.add_argument('--save', type=str, help='Save processes to file')
    parser.add_argument('--fcfs', action='store_true', help='Run FCFS algorithm')
    parser.add_argument('--sjf', action='store_true', help='Run SJF algorithm')
    parser.add_argument('--priority', action='store_true', help='Run Priority algorithm')
    parser.add_argument('--rr', action='store_true', help='Run Round Robin algorithm')
    parser.add_argument('--prr', action='store_true', help='Run Priority Round Robin algorithm')
    parser.add_argument('--mfq', action='store_true', help='Run Multilevel Feedback Queue algorithm')
    parser.add_argument('--all', action='store_true', help='Run all algorithms')
    parser.add_argument('--quantum', type=int, default=get_config("round_robin", "default_quantum"), 
                        help='Time quantum for Round Robin')
    parser.add_argument('--gantt', action='store_true', help='Show Gantt chart')
    parser.add_argument('--timeline', action='store_true', help='Show process timeline')
    parser.add_argument('--compare', action='store_true', help='Compare algorithms')
    parser.add_argument('--advanced', action='store_true', help='Use advanced visualizations')
    parser.add_argument('--output', type=str, help='Output directory for results')
    parser.add_argument('--dashboard', action='store_true', help='Create comprehensive dashboard')
    parser.add_argument('--show', action='store_true', help='Show visualizations')
    
    args = parser.parse_args()
    
    # Generate or load processes
    if args.random:
        processes = generate_random_processes(args.random)
    elif args.file:
        processes = load_processes_from_file(args.file)
    else:
        processes = generate_random_processes(get_config("random_process", "default_count"))
        
    # Save processes if requested
    if args.save:
        save_processes_to_file(processes, args.save)
    
    # Set up algorithms to run
    algorithms = {}
    if args.fcfs or args.all:
        algorithms["First Come First Serve"] = FCFSScheduler
    if args.sjf or args.all:
        algorithms["Shortest Job First"] = SJFScheduler
    if args.priority or args.all:
        algorithms["Priority Scheduling"] = PriorityScheduler
    if args.rr or args.all:
        algorithms["Round Robin"] = lambda p: RoundRobinScheduler(p, args.quantum)
    if args.prr or args.all:
        algorithms["Priority Round Robin"] = lambda p: PriorityRRScheduler(p, args.quantum)
    if args.mfq or args.all:
        mfq_config = get_config("multilevel_feedback_queue")
        algorithms["Multilevel Feedback Queue"] = lambda p: MFQScheduler(
            p, mfq_config["num_queues"], mfq_config["base_quantum"]
        )
    
    # If no algorithms specified, default to all
    if not algorithms:
        args.all = True
        algorithms["First Come First Serve"] = FCFSScheduler
        algorithms["Shortest Job First"] = SJFScheduler
        algorithms["Priority Scheduling"] = PriorityScheduler
        algorithms["Round Robin"] = lambda p: RoundRobinScheduler(p, args.quantum)
        algorithms["Priority Round Robin"] = lambda p: PriorityRRScheduler(p, args.quantum)
        mfq_config = get_config("multilevel_feedback_queue")
        algorithms["Multilevel Feedback Queue"] = lambda p: MFQScheduler(
            p, mfq_config["num_queues"], mfq_config["base_quantum"]
        )
    
    # Run simulation
    output_dir = args.output if args.output else None
    results = run_simulation(processes, algorithms, output_dir)
    
    # Prepare visualizations based on arguments
    if args.gantt or args.timeline or args.compare or args.show or args.dashboard or args.advanced:
        if args.output:
            vis_dir = os.path.join(args.output, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
        else:
            vis_dir = None
        
        # Use advanced visualizations if requested
        if args.advanced or args.dashboard:
            if args.dashboard:
                # Create full dashboard with all advanced visualizations
                dashboard_dir = os.path.join(vis_dir, 'dashboard') if vis_dir else None
                advanced_visualizations.create_comparative_dashboard(
                    results, processes, "CPU Scheduler Comparison", dashboard_dir
                )
                if args.show:
                    print("\nDashboard created. Open the files in the dashboard directory to view.")
            else:
                # Create individual advanced visualizations
                if args.gantt or args.show:
                    for alg_name, result in results.items():
                        if 'execution_sequence' in result:
                            filename = os.path.join(vis_dir, f"{alg_name}_lifecycle.png") if vis_dir else None
                            advanced_visualizations.create_process_lifecycle_visualization(
                                result['execution_sequence'], processes, alg_name, filename
                            )
                            if filename:
                                print(f"Saved lifecycle visualization for {alg_name} to {filename}")
                
                if args.compare or args.show:
                    metrics_list = ['avg_waiting_time', 'avg_turnaround_time', 'avg_response_time', 'throughput', 'cpu_utilization']
                    heatmap_file = os.path.join(vis_dir, "metrics_heatmap.png") if vis_dir else None
                    advanced_visualizations.create_heatmap(
                        results, metrics_list, "Algorithm Performance Comparison", heatmap_file
                    )
                    if heatmap_file:
                        print(f"Saved metrics heatmap to {heatmap_file}")
                    
                    radar_file = os.path.join(vis_dir, "metrics_radar.png") if vis_dir else None
                    all_metrics = set()
                    for alg, metrics_dict in results.items():
                        all_metrics.update([m for m in metrics_dict.keys() if isinstance(metrics_dict[m], (int, float)) and 
                                        m not in ('execution_sequence', 'completed_processes')])
                    advanced_visualizations.create_radar_chart(
                        results, list(all_metrics), "Algorithm Metrics Comparison", radar_file
                    )
                    if radar_file:
                        print(f"Saved metrics radar chart to {radar_file}")
        else:
            # Use standard visualizations
            if args.gantt or args.show:
                for alg_name, result in results.items():
                    if 'execution_sequence' in result:
                        filename = os.path.join(vis_dir, f"{alg_name}_gantt.png") if vis_dir else None
                        visualization.create_gantt_chart(result['execution_sequence'], alg_name, filename)
                        if filename:
                            print(f"Saved Gantt chart for {alg_name} to {filename}")
            
            if args.timeline or args.show:
                for alg_name, result in results.items():
                    if 'execution_sequence' in result:
                        filename = os.path.join(vis_dir, f"{alg_name}_timeline.png") if vis_dir else None
                        visualization.create_process_timeline(result['execution_sequence'], processes, alg_name, filename)
                        if filename:
                            print(f"Saved process timeline for {alg_name} to {filename}")
            
            if args.compare or args.show:
                filename = os.path.join(vis_dir, "algorithm_comparison.png") if vis_dir else None
                visualization.compare_algorithms_chart(results, filename)
                if filename:
                    print(f"Saved algorithm comparison chart to {filename}")

if __name__ == "__main__":
    main()
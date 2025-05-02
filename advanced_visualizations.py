import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os

def create_heatmap(results, metrics, title, filename=None):
    """
    Create a heatmap comparing algorithms across multiple metrics.
    
    Args:
        results: Dictionary mapping algorithm names to their metrics
        metrics: List of metrics to include
        title: Title for the heatmap
        filename: Optional filename to save the plot
        
    Returns:
        The matplotlib figure object
    """
    # Create a DataFrame for the heatmap
    algorithms = list(results.keys())
    data = []
    
    for metric in metrics:
        row = []
        for alg in algorithms:
            if metric in results[alg]:
                row.append(results[alg][metric])
            else:
                row.append(np.nan)
        data.append(row)
    
    df = pd.DataFrame(data, index=metrics, columns=algorithms)
    
    # Normalize each metric to 0-1 scale for fair comparison
    normalized_df = df.copy()
    for metric in metrics:
        if metric in ['context_switches']:  # Lower is better
            normalized_df.loc[metric] = 1 - (df.loc[metric] - df.loc[metric].min()) / (df.loc[metric].max() - df.loc[metric].min() + 1e-10)
        else:  # Higher is better
            normalized_df.loc[metric] = (df.loc[metric] - df.loc[metric].min()) / (df.loc[metric].max() - df.loc[metric].min() + 1e-10)
    
    # Create the plot
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(normalized_df, annot=df, fmt=".2f", cmap="YlGnBu", linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
        plt.close()
        return None
    else:
        return fig

def create_radar_chart(results, metrics, title, filename=None):
    """
    Create a radar chart comparing algorithms across multiple metrics.
    
    Args:
        results: Dictionary mapping algorithm names to their metrics
        metrics: List of metrics to include
        title: Title for the radar chart
        filename: Optional filename to save the plot
        
    Returns:
        The matplotlib figure object
    """
    algorithms = list(results.keys())
    num_metrics = len(metrics)
    
    # Normalize the data for radar chart
    normalized_data = {}
    for metric in metrics:
        min_val = min(results[alg].get(metric, 0) for alg in algorithms)
        max_val = max(results[alg].get(metric, 0) for alg in algorithms)
        range_val = max_val - min_val if max_val > min_val else 1
        
        for alg in algorithms:
            if alg not in normalized_data:
                normalized_data[alg] = []
            
            if metric in results[alg]:
                if metric in ['context_switches']:  # Lower is better
                    val = 1 - (results[alg][metric] - min_val) / range_val
                else:  # Higher is better
                    val = (results[alg][metric] - min_val) / range_val
                normalized_data[alg].append(val)
            else:
                normalized_data[alg].append(0)
    
    # Set up the radar chart
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Add metrics labels
    plt.xticks(angles[:-1], metrics, size=12)
    
    # Plot each algorithm
    for alg in algorithms:
        values = normalized_data[alg]
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, label=alg)
        ax.fill(angles, values, alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(title, size=15)
    
    if filename:
        plt.savefig(filename)
        plt.close()
        return None
    else:
        return fig

def create_process_lifecycle_visualization(execution_sequence, processes, title, filename=None):
    """
    Create an advanced visualization showing process lifecycles.
    
    Args:
        execution_sequence: List of execution segments from a scheduler
        processes: List of Process objects
        title: Title for the visualization
        filename: Optional filename to save the plot
        
    Returns:
        The matplotlib figure object
    """
    # Create a color map for processes
    num_processes = len(processes)
    colors = plt.cm.viridis(np.linspace(0, 0.9, num_processes))
    process_colors = {p.pid: colors[i] for i, p in enumerate(processes)}
    
    # Sort processes by arrival time
    sorted_processes = sorted(processes, key=lambda p: p.arrival_time)
    
    # Find the last completion time
    max_time = max(seg['end'] for seg in execution_sequence)
    
    # Create figure with GridSpec for layout
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 1, height_ratios=[2, 1, 1])
    
    # 1. Gantt chart
    ax_gantt = fig.add_subplot(gs[0])
    ax_gantt.set_title(f"Process Execution Timeline: {title}")
    ax_gantt.set_xlabel("Time")
    ax_gantt.set_ylabel("Process ID")
    
    # Plot execution blocks
    for segment in execution_sequence:
        pid = segment['pid']
        start = segment['start']
        end = segment['end']
        duration = end - start
        
        ax_gantt.barh(pid, duration, left=start, color=process_colors[pid], 
                     edgecolor='black', alpha=0.7)
        
        # Add labels on longer segments
        if duration >= 2:
            ax_gantt.text(start + duration/2, pid, f"P{pid}", 
                         ha='center', va='center', color='black')
    
    # Set y-ticks to process IDs
    pids = sorted([p.pid for p in processes])
    ax_gantt.set_yticks(pids)
    ax_gantt.set_yticklabels([f"P{pid}" for pid in pids])
    ax_gantt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 2. Process States
    ax_states = fig.add_subplot(gs[1], sharex=ax_gantt)
    ax_states.set_title("Process States Over Time")
    ax_states.set_ylabel("Process ID")
    
    # For each process, determine its state at each point in time
    time_points = sorted(list(set([0] + 
                                  [seg['start'] for seg in execution_sequence] + 
                                  [seg['end'] for seg in execution_sequence])))
    
    for p in sorted_processes:
        states = []
        current_state = "Not Arrived" if p.arrival_time > 0 else "Ready"
        
        for t in range(int(max_time) + 1):
            # Check if process arrived
            if t == p.arrival_time:
                current_state = "Ready"
                
            # Check if process is running at time t
            is_running = False
            for seg in execution_sequence:
                if seg['pid'] == p.pid and seg['start'] <= t < seg['end']:
                    current_state = "Running"
                    is_running = True
                    break
            
            # If not running but arrived, and not completed, it's waiting
            if not is_running and t >= p.arrival_time:
                if p.completion_time and t >= p.completion_time:
                    current_state = "Completed"
                elif current_state == "Running":
                    current_state = "Ready"
            
            states.append(current_state)
        
        # Plot states as colored segments
        state_colors = {
            "Not Arrived": "white",
            "Ready": "orange",
            "Running": "green",
            "Completed": "blue"
        }
        
        for t in range(len(states)-1):
            state = states[t]
            ax_states.barh(p.pid, 1, left=t, color=state_colors[state], 
                          edgecolor=None, alpha=0.7)
    
    # Create legend for states
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="white", edgecolor='gray', label='Not Arrived'),
        Patch(facecolor="orange", edgecolor='gray', label='Ready/Waiting'),
        Patch(facecolor="green", edgecolor='gray', label='Running'),
        Patch(facecolor="blue", edgecolor='gray', label='Completed')
    ]
    ax_states.legend(handles=legend_elements, loc='upper right')
    
    # Set y-ticks to process IDs
    ax_states.set_yticks(pids)
    ax_states.set_yticklabels([f"P{pid}" for pid in pids])
    
    # 3. CPU Utilization
    ax_util = fig.add_subplot(gs[2], sharex=ax_gantt)
    ax_util.set_title("CPU Utilization")
    ax_util.set_xlabel("Time")
    ax_util.set_ylabel("CPU Status")
    
    # Determine CPU utilization at each time point
    cpu_status = []
    for t in range(int(max_time) + 1):
        is_cpu_busy = False
        for seg in execution_sequence:
            if seg['start'] <= t < seg['end']:
                is_cpu_busy = True
                break
        cpu_status.append(1 if is_cpu_busy else 0)
    
    # Plot CPU utilization
    ax_util.step(range(len(cpu_status)), cpu_status, where='post', color='red')
    ax_util.fill_between(range(len(cpu_status)), cpu_status, step='post', alpha=0.3, color='red')
    ax_util.set_yticks([0, 1])
    ax_util.set_yticklabels(['Idle', 'Busy'])
    
    # Calculate and display CPU utilization percentage
    utilization = sum(cpu_status) / len(cpu_status) * 100
    ax_util.text(max_time/2, 0.5, f"Utilization: {utilization:.1f}%", 
                ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
        plt.close()
        return None
    else:
        return fig

def create_comparative_dashboard(results, processes, title="Algorithm Comparison Dashboard", output_dir=None):
    """
    Create a comprehensive visual dashboard comparing multiple scheduling algorithms.
    
    Args:
        results: Dictionary mapping algorithm names to their results
        processes: List of Process objects used in the simulation
        title: Title for the dashboard
        output_dir: Directory to save the dashboard components
        
    Returns:
        Dictionary with figure objects or file paths
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract common metrics for comparison
    common_metrics = ['avg_waiting_time', 'avg_turnaround_time', 'avg_response_time', 'throughput', 'cpu_utilization']
    specialized_metrics = ['context_switches', 'fairness', 'priority_preference_ratio', 'quantum_utilization', 'avg_queue_level']
    
    # 1. Generate heatmap for common metrics
    heatmap_file = os.path.join(output_dir, "metrics_heatmap.png") if output_dir else None
    heatmap_fig = create_heatmap(results, common_metrics, f"{title} - Common Metrics", heatmap_file)
    
    # 2. Generate radar chart for all available metrics
    all_metrics = set()
    for alg, metrics in results.items():
        all_metrics.update([m for m in metrics.keys() if isinstance(metrics[m], (int, float)) and 
                           m not in ('execution_sequence', 'completed_processes')])
    
    radar_file = os.path.join(output_dir, "metrics_radar.png") if output_dir else None
    radar_fig = create_radar_chart(results, list(all_metrics), f"{title} - All Metrics", radar_file)
    
    # 3. Generate process lifecycle visualizations for each algorithm
    lifecycle_figs = {}
    for alg, result in results.items():
        if 'execution_sequence' in result:
            lifecycle_file = os.path.join(output_dir, f"{alg.replace(' ', '_')}_lifecycle.png") if output_dir else None
            lifecycle_fig = create_process_lifecycle_visualization(result['execution_sequence'], processes, alg, lifecycle_file)
            if not output_dir:  # Only store figures if not saving to files
                lifecycle_figs[alg] = lifecycle_fig
    
    # Generate summarized metrics table
    print("=" * 50)
    print(f"{title} - Summary")
    print("=" * 50)
    
    algorithms = list(results.keys())
    metrics_to_show = [m for m in all_metrics if any(m in results[alg] for alg in algorithms)]
    
    header = f"{'Metric':<25} " + " ".join([f"{alg:<15}" for alg in algorithms])
    print(header)
    print("-" * len(header))
    
    for metric in sorted(metrics_to_show):
        values = []
        for alg in algorithms:
            if metric in results[alg]:
                values.append(f"{results[alg][metric]:<15.2f}")
            else:
                values.append(f"{'N/A':<15}")
        
        print(f"{metric:<25} " + " ".join(values))
    
    print("\n")
    
    if output_dir:
        print(f"Dashboard visualizations saved to: {output_dir}")
        return {
            "heatmap": heatmap_file,
            "radar": radar_file,
            "lifecycles": [os.path.join(output_dir, f"{alg.replace(' ', '_')}_lifecycle.png") 
                          for alg in algorithms if 'execution_sequence' in results[alg]]
        }
    else:
        return {
            "heatmap": heatmap_fig,
            "radar": radar_fig,
            "lifecycles": lifecycle_figs
        }
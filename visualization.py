import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import os
from datetime import datetime

def create_gantt_chart(execution_sequence, algorithm_name, save_file=None):
    """
    Create an enhanced Gantt chart of the scheduling sequence.
    
    Args:
        execution_sequence: List of execution segments with pid, start, and end times
        algorithm_name: Name of the scheduling algorithm
        save_file: Path to save the chart, if None the chart is displayed
        
    Returns:
        The matplotlib figure object
    """
    if not execution_sequence:
        print("No execution sequence to visualize")
        return None
    
    # Create figure and axis with improved size and resolution
    fig = plt.figure(figsize=(14, 7), dpi=100)
    
    # Define a vibrant color palette
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(set(item['pid'] for item in execution_sequence))))
    
    # Map process IDs to colors
    process_ids = sorted(set(item['pid'] for item in execution_sequence))
    color_map = {pid: colors[i] for i, pid in enumerate(process_ids)}
    
    # Create subplots: Gantt chart and legend
    gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1])
    ax_gantt = plt.subplot(gs[0])
    ax_legend = plt.subplot(gs[1])
    
    # Create Gantt chart
    y_ticks = []
    y_labels = []
    
    for i, pid in enumerate(process_ids):
        y_ticks.append(i)
        y_labels.append(f"Process {pid}")
        
        # Plot segments for this process
        for item in execution_sequence:
            if item['pid'] == pid:
                ax_gantt.barh(
                    i, 
                    item['end'] - item['start'], 
                    left=item['start'], 
                    color=color_map[pid],
                    edgecolor='black',
                    alpha=0.8,
                    height=0.6
                )
                
                # Add text label for duration
                duration = item['end'] - item['start']
                if duration > 1:  # Only add text if bar is wide enough
                    ax_gantt.text(
                        item['start'] + duration/2, 
                        i, 
                        str(duration),
                        ha='center', 
                        va='center',
                        color='white' if duration > 2 else 'black',
                        fontweight='bold',
                        fontsize=9
                    )
    
    # Customize Gantt chart
    ax_gantt.set_yticks(y_ticks)
    ax_gantt.set_yticklabels(y_labels)
    ax_gantt.set_xlabel("Time Units", fontweight='bold')
    ax_gantt.set_title(f"Process Execution Timeline - {algorithm_name}", 
                       fontweight='bold', fontsize=14)
    
    # Add grid for better readability
    ax_gantt.grid(True, axis='x', linestyle='--', alpha=0.7)
    ax_gantt.set_axisbelow(True)
    
    # Set y-axis limits for better spacing
    ax_gantt.set_ylim(-0.5, len(process_ids) - 0.5)
    
    # Add vertical lines at integer time points
    max_time = max(item['end'] for item in execution_sequence)
    for t in range(0, int(max_time) + 1):
        ax_gantt.axvline(x=t, color='gray', linestyle='-', alpha=0.2)
    
    # Create custom legend
    legend_elements = [
        Patch(facecolor=color_map[pid], edgecolor='black', label=f"Process {pid}")
        for pid in process_ids
    ]
    ax_legend.legend(handles=legend_elements, loc='center', frameon=True)
    ax_legend.axis('off')
    
    # Add algorithm info and timestamp
    algorithm_info = f"Algorithm: {algorithm_name}\n"
    algorithm_info += f"Total Processes: {len(process_ids)}\n"
    algorithm_info += f"Total Time: {max_time} units\n"
    algorithm_info += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    plt.figtext(0.75, 0.15, algorithm_info, ha="center", fontsize=9,
                bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if needed
    if save_file:
        plt.savefig(save_file, bbox_inches='tight')
        print(f"Gantt chart saved to {save_file}")
    
    # Return the figure instead of showing it
    return fig

def create_metrics_comparison_chart(comparison_results, save_file=None):
    """
    Create enhanced bar charts to compare metrics across algorithms.
    
    Args:
        comparison_results: Dictionary with metrics as keys and algorithm results as values
        save_file: Path to save the chart, if None the chart is displayed
        
    Returns:
        The matplotlib figure object
    """
    if not comparison_results:
        print("No comparison results to visualize")
        return None
    
    # Get algorithms and metrics
    algorithms = list(comparison_results.keys())
    
    # Find common metrics across all algorithms
    metrics = set()
    for alg, alg_metrics in comparison_results.items():
        metrics.update([m for m in alg_metrics.keys() if isinstance(alg_metrics[m], (int, float))])
    
    # Group metrics into categories
    time_metrics = [m for m in metrics if 'time' in m]
    utilization_metrics = [m for m in metrics if 'utilization' in m or 'throughput' in m]
    other_metrics = [m for m in metrics if m not in time_metrics and m not in utilization_metrics]
    
    # Create a figure with appropriate size
    n_metrics = len(metrics)
    fig = plt.figure(figsize=(12, max(3 * len(time_metrics), 10)), dpi=100)
    
    # Create gridspec for layout
    gs = gridspec.GridSpec(3, 1, height_ratios=[
        max(1, len(time_metrics)), 
        max(1, len(utilization_metrics)), 
        max(1, len(other_metrics))
    ])
    
    # Create subplots for each category
    ax_time = plt.subplot(gs[0]) if time_metrics else None
    ax_utilization = plt.subplot(gs[1]) if utilization_metrics else None
    ax_other = plt.subplot(gs[2]) if other_metrics else None
    
    # Plot time metrics
    if time_metrics:
        for i, metric in enumerate(time_metrics):
            # Extract values for each algorithm
            algs = []
            values = []
            for alg in algorithms:
                if metric in comparison_results[alg]:
                    algs.append(alg)
                    values.append(comparison_results[alg][metric])
            
            # Create position offset for grouped bars
            width = 0.8 / len(time_metrics)
            positions = np.arange(len(algs)) + (i - len(time_metrics)/2 + 0.5) * width
            
            # Plot bars
            ax_time.bar(positions, values, width=width, label=metric)
            
        ax_time.set_title("Time Metrics", fontweight='bold')
        ax_time.set_xticks(np.arange(len(algs)))
        ax_time.set_xticklabels(algs, rotation=45, ha='right')
        ax_time.legend()
        ax_time.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Plot utilization metrics
    if utilization_metrics:
        for i, metric in enumerate(utilization_metrics):
            # Extract values for each algorithm
            algs = []
            values = []
            for alg in algorithms:
                if metric in comparison_results[alg]:
                    algs.append(alg)
                    values.append(comparison_results[alg][metric])
            
            # Create position offset for grouped bars
            width = 0.8 / len(utilization_metrics)
            positions = np.arange(len(algs)) + (i - len(utilization_metrics)/2 + 0.5) * width
            
            # Plot bars
            ax_utilization.bar(positions, values, width=width, label=metric)
            
        ax_utilization.set_title("Utilization Metrics", fontweight='bold')
        ax_utilization.set_xticks(np.arange(len(algs)))
        ax_utilization.set_xticklabels(algs, rotation=45, ha='right')
        ax_utilization.legend()
        ax_utilization.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Plot other metrics
    if other_metrics:
        for i, metric in enumerate(other_metrics):
            # Extract values for each algorithm
            algs = []
            values = []
            for alg in algorithms:
                if metric in comparison_results[alg]:
                    algs.append(alg)
                    values.append(comparison_results[alg][metric])
            
            # Create position offset for grouped bars
            width = 0.8 / len(other_metrics)
            positions = np.arange(len(algs)) + (i - len(other_metrics)/2 + 0.5) * width
            
            # Plot bars
            ax_other.bar(positions, values, width=width, label=metric)
            
        ax_other.set_title("Other Metrics", fontweight='bold')
        ax_other.set_xticks(np.arange(len(algs)))
        ax_other.set_xticklabels(algs, rotation=45, ha='right')
        ax_other.legend()
        ax_other.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if needed
    if save_file:
        plt.savefig(save_file, bbox_inches='tight')
        print(f"Metrics comparison chart saved to {save_file}")
    
    # Return the figure
    return fig

def create_waiting_time_by_priority_chart(priority_data, algorithm_name, save_file=None):
    """
    Create a bar chart showing waiting time by priority level.
    
    Args:
        priority_data: Dictionary mapping priority levels to waiting times
        algorithm_name: Name of the scheduling algorithm
        save_file: Path to save the chart, if None the chart is displayed
        
    Returns:
        The matplotlib figure object
    """
    if not priority_data:
        print("No priority data to visualize")
        return None
    
    # Create figure
    fig = plt.figure(figsize=(10, 6), dpi=100)
    
    # Sort priorities (lower number = higher priority)
    priorities = sorted(priority_data.keys())
    waiting_times = [priority_data[p] for p in priorities]
    
    # Create bars with gradient coloring (red for high priority, green for low)
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(priorities)))
    bars = plt.bar(priorities, waiting_times, color=colors, alpha=0.8)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.1,
            f'{height:.2f}',
            ha='center', 
            va='bottom'
        )
    
    # Set labels and title
    plt.xlabel('Priority Level (Lower = Higher Priority)', fontweight='bold')
    plt.ylabel('Average Waiting Time', fontweight='bold')
    plt.title(f'Waiting Time by Priority Level - {algorithm_name}', 
             fontweight='bold', fontsize=14)
    
    # Add grid
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.gca().set_axisbelow(True)
    
    # Set y
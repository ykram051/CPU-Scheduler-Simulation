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
    """
    if not execution_sequence:
        print("No execution sequence to visualize")
        return
    
    # Get unique process IDs and sort them
    process_ids = sorted(set(item['pid'] for item in execution_sequence))
    
    # Create figure and axis with improved size and resolution
    plt.figure(figsize=(14, 7), dpi=100)
    
    # Define a vibrant color palette
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(process_ids)))
    
    # Map process IDs to colors
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
    
    # Save or show
    if save_file:
        plt.savefig(save_file, bbox_inches='tight')
        print(f"Gantt chart saved to {save_file}")
    else:
        plt.show()
    
    plt.close()

def create_metrics_comparison_chart(comparison_results, save_file=None):
    """
    Create enhanced bar charts to compare metrics across algorithms.
    
    Args:
        comparison_results: Dictionary with metric comparisons
        save_file: Path to save the chart, if None the chart is displayed
    """
    if not comparison_results:
        print("No comparison results to visualize")
        return
    
    # Get metrics and algorithms
    metrics = list(comparison_results.keys())
    
    # Filter out complex metrics (dictionaries)
    metrics = [m for m in metrics if isinstance(
        list(comparison_results[m]['values'].values())[0], 
        (int, float)
    )]
    
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
    for i, (title, category_metrics) in enumerate([
        ("Time-based Metrics", time_metrics),
        ("Utilization Metrics", utilization_metrics),
        ("Other Performance Metrics", other_metrics)
    ]):
        if not category_metrics:
            continue
            
        ax = plt.subplot(gs[i])
        
        # Position multiple metrics in this category
        positions = np.arange(0, len(category_metrics) * 4, 4)
        width = 0.7
        
        algorithms = list(comparison_results[category_metrics[0]]['values'].keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
        
        # For each algorithm, plot its value for each metric
        for j, alg in enumerate(algorithms):
            values = [comparison_results[metric]['values'].get(alg, 0) for metric in category_metrics]
            bars = ax.bar(positions + j*width, values, width=width, 
                          color=colors[j], label=alg, alpha=0.8)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.05 * max(values),
                    f'{height:.2f}',
                    ha='center', 
                    va='bottom',
                    fontsize=8
                )
        
        # Highlight best algorithm for each metric
        for k, metric in enumerate(category_metrics):
            best_alg = comparison_results[metric]['best_algorithm']
            best_idx = algorithms.index(best_alg)
            
            # Draw a star on the best algorithm bar
            best_value = comparison_results[metric]['values'][best_alg]
            ax.plot(positions[k] + best_idx*width + width/2, best_value + 0.15 * max(values), 
                    marker='*', markersize=12, color='gold', markeredgecolor='black')
        
        # Set x ticks at the center of each metric group
        ax.set_xticks(positions + width * (len(algorithms) - 1) / 2)
        ax.set_xticklabels([metric.replace('_', ' ').title() for metric in category_metrics])
        
        # Set title and grid
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
    
    # Add legend at the bottom
    plt.figlegend(
        loc='lower center', 
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(5, len(algorithms))
    )
    
    # Add overall title
    plt.suptitle('CPU Scheduling Algorithm Performance Comparison', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save or show
    if save_file:
        plt.savefig(save_file, bbox_inches='tight')
        print(f"Comparison chart saved to {save_file}")
    else:
        plt.show()
    
    plt.close()

def create_process_timeline(processes, algorithm_name, save_file=None):
    """
    Create a timeline visualization showing the lifecycle of each process.
    
    Args:
        processes: List of completed processes
        algorithm_name: Name of the scheduling algorithm
        save_file: Path to save the chart, if None the chart is displayed
    """
    if not processes:
        print("No processes to visualize")
        return
    
    # Sort processes by arrival time
    processes = sorted(processes, key=lambda p: p.arrival_time)
    
    # Create figure
    plt.figure(figsize=(14, 8), dpi=100)
    
    # Define colors for different states
    colors = {
        'waiting': 'lightgray',
        'running': 'dodgerblue',
        'complete': 'forestgreen'
    }
    
    # Get the maximum completion time
    max_time = max(p.completion_time for p in processes)
    
    # Create a subplot for each process
    for i, process in enumerate(processes):
        plt.subplot(len(processes), 1, i+1)
        
        # Calculate waiting time periods
        total_execution_time = 0
        waiting_time = process.waiting_time
        
        # Plot waiting time (from arrival to start)
        plt.barh(
            0, 
            process.start_time - process.arrival_time, 
            left=process.arrival_time,
            color=colors['waiting'],
            alpha=0.7,
            label='Waiting' if i == 0 else ""
        )
        
        # Plot execution time
        plt.barh(
            0,
            process.burst_time,
            left=process.start_time,
            color=colors['running'],
            alpha=0.7,
            label='Running' if i == 0 else ""
        )
        
        # Add markers for key events
        plt.plot(process.arrival_time, 0, 'v', color='red', markersize=7, label='Arrival' if i == 0 else "")
        plt.plot(process.start_time, 0, '>', color='blue', markersize=7, label='Start' if i == 0 else "")
        plt.plot(process.completion_time, 0, 'D', color='green', markersize=7, label='Completion' if i == 0 else "")
        
        # Add text annotations
        plt.text(process.arrival_time, 0.15, f"A:{process.arrival_time}", ha='center', va='bottom', fontsize=8)
        plt.text(process.completion_time, 0.15, f"C:{process.completion_time}", ha='center', va='bottom', fontsize=8)
        
        # Add process info
        plt.text(
            0, 
            0, 
            f"P{process.pid} (Burst:{process.burst_time}, Priority:{process.priority})",
            ha='right', 
            va='center',
            fontweight='bold',
            fontsize=9
        )
        
        # Add turnaround and waiting time
        plt.text(
            max_time + 1, 
            0, 
            f"TAT:{process.turnaround_time}, WT:{process.waiting_time}",
            ha='left', 
            va='center',
            fontsize=9
        )
        
        # Remove y ticks and labels
        plt.yticks([])
        
        # Set x axis limits
        plt.xlim(-3, max_time + 5)
        
        # Only show x axis labels for the bottom subplot
        if i == len(processes) - 1:
            plt.xlabel("Time Units", fontweight='bold')
        else:
            plt.tick_params(labelbottom=False)
    
    # Add legend on the top
    plt.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, len(processes) + 0.3),
        ncol=5,
        frameon=True
    )
    
    # Add title
    plt.suptitle(
        f'Process Timeline - {algorithm_name}',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save or show
    if save_file:
        plt.savefig(save_file, bbox_inches='tight')
        print(f"Process timeline saved to {save_file}")
    else:
        plt.show()
    
    plt.close()

def create_comprehensive_report(results, save_dir=None):
    """
    Create a comprehensive visualization report for all algorithms.
    
    Args:
        results: Dictionary containing results for each algorithm
        save_dir: Directory to save visualizations, if None they are displayed
    """
    if not results:
        print("No results to visualize")
        return
        
    # Ensure directory exists
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create individual Gantt charts for each algorithm
    for algorithm_name, result in results.items():
        if 'execution_sequence' in result:
            gantt_file = None
            if save_dir:
                gantt_file = os.path.join(save_dir, f"{algorithm_name.replace(' ', '_').lower()}_gantt.png")
            
            create_gantt_chart(
                result['execution_sequence'],
                algorithm_name,
                save_file=gantt_file
            )
        
        # Create process timeline if completed_processes are available
        if 'completed_processes' in result:
            timeline_file = None
            if save_dir:
                timeline_file = os.path.join(save_dir, f"{algorithm_name.replace(' ', '_').lower()}_timeline.png")
            
            create_process_timeline(
                result['completed_processes'],
                algorithm_name,
                save_file=timeline_file
            )
    
    # Create comparison charts
    if len(results) > 1:
        # Prepare comparison data (omitting non-metric keys)
        comparison_data = {}
        for alg, result in results.items():
            comparison_data[alg] = {
                k: v for k, v in result.items() 
                if k not in ['execution_sequence', 'completed_processes'] and not isinstance(v, dict)
            }
        
        # Generate comparison results
        from metrics import compare_algorithms
        comparison = compare_algorithms(comparison_data)
        
        # Create comparison chart
        comparison_file = None
        if save_dir:
            comparison_file = os.path.join(save_dir, "algorithm_comparison.png")
        
        create_metrics_comparison_chart(
            comparison,
            save_file=comparison_file
        )

def create_waiting_time_by_priority_chart(priority_data, algorithm_name, save_file=None):
    """
    Create a bar chart showing waiting time by priority level.
    
    Args:
        priority_data: Dictionary mapping priority levels to waiting times
        algorithm_name: Name of the scheduling algorithm
        save_file: Path to save the chart, if None the chart is displayed
    """
    if not priority_data:
        print("No priority data to visualize")
        return
    
    # Create figure
    plt.figure(figsize=(10, 6), dpi=100)
    
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
    
    # Set y-axis to start at zero
    plt.ylim(bottom=0)
    
    # Set x-ticks to include all priority levels
    plt.xticks(priorities)
    
    # Add text explanation
    plt.figtext(
        0.5, 0.01, 
        "Note: Lower priority numbers represent higher priority processes",
        ha='center', fontsize=9, style='italic'
    )
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Save or show
    if save_file:
        plt.savefig(save_file, bbox_inches='tight')
        print(f"Priority waiting time chart saved to {save_file}")
    else:
        plt.show()
    
    plt.close()
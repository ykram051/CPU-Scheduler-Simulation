def calculate_basic_metrics(completed_processes):
    """Calculate basic metrics common to all scheduling algorithms"""
    if not completed_processes:
        return {
            'avg_turnaround_time': 0,
            'avg_waiting_time': 0,
            'avg_response_time': 0,
            'throughput': 0,
            'cpu_utilization': 0
        }
    
    # Calculate turnaround time (completion - arrival)
    total_turnaround_time = 0
    for process in completed_processes:
        process.turnaround_time = process.completion_time - process.arrival_time
        total_turnaround_time += process.turnaround_time
    avg_turnaround_time = total_turnaround_time / len(completed_processes)
    
    # Calculate waiting time (turnaround - burst)
    total_waiting_time = 0
    for process in completed_processes:
        process.waiting_time = process.turnaround_time - process.burst_time
        total_waiting_time += process.waiting_time
    avg_waiting_time = total_waiting_time / len(completed_processes)
    
    # Calculate response time (first CPU - arrival)
    total_response_time = 0
    for process in completed_processes:
        process.response_time = process.start_time - process.arrival_time
        total_response_time += process.response_time
    avg_response_time = total_response_time / len(completed_processes)
    
    # Calculate throughput
    max_completion_time = max(p.completion_time for p in completed_processes)
    throughput = len(completed_processes) / max_completion_time if max_completion_time > 0 else 0
    
    # Calculate CPU utilization
    total_burst_time = sum(p.burst_time for p in completed_processes)
    cpu_utilization = (total_burst_time / max_completion_time) * 100 if max_completion_time > 0 else 0
    
    return {
        'avg_turnaround_time': avg_turnaround_time,
        'avg_waiting_time': avg_waiting_time,
        'avg_response_time': avg_response_time,
        'throughput': throughput,
        'cpu_utilization': cpu_utilization
    }

def calculate_fcfs_metrics(completed_processes, execution_sequence):
    """Calculate metrics for FCFS scheduling"""
    metrics = calculate_basic_metrics(completed_processes)
    
    # For FCFS, we can verify that processes are executed in order of arrival
    is_fcfs_order = True
    if len(completed_processes) > 1:
        sorted_processes = sorted(completed_processes, key=lambda p: p.arrival_time)
        is_fcfs_order = all(sorted_processes[i].start_time <= sorted_processes[i+1].start_time 
                         for i in range(len(sorted_processes)-1))
    
    metrics['is_fcfs_order'] = is_fcfs_order
    return metrics

def calculate_sjf_metrics(completed_processes, execution_sequence):
    """Calculate metrics for SJF scheduling"""
    metrics = calculate_basic_metrics(completed_processes)
    
    # For SJF, we can check if shorter processes are executed before longer ones when available
    available_processes = {}
    current_time = 0
    is_sjf_optimal = True
    
    # Sort processes by start time to analyze execution order
    sorted_by_start = sorted(completed_processes, key=lambda p: p.start_time)
    
    for process in sorted_by_start:
        # Update available processes at this time
        arrival_time = process.arrival_time
        if arrival_time <= current_time and process not in available_processes:
            available_processes[process] = process.burst_time
            
        # Check if this was the shortest job among available
        if available_processes and min(available_processes.values()) < process.burst_time:
            is_sjf_optimal = False
            break
            
        # Update time and remove the executed process
        current_time = process.completion_time
        if process in available_processes:
            del available_processes[process]
    
    metrics['is_sjf_optimal'] = is_sjf_optimal
    return metrics

def calculate_priority_metrics(completed_processes, execution_sequence):
    """Calculate metrics for Priority scheduling"""
    metrics = calculate_basic_metrics(completed_processes)
    
    # Calculate waiting time by priority level
    priority_waiting_times = {}
    priority_counts = {}
    
    for process in completed_processes:
        if process.priority not in priority_waiting_times:
            priority_waiting_times[process.priority] = 0
            priority_counts[process.priority] = 0
            
        priority_waiting_times[process.priority] += process.waiting_time
        priority_counts[process.priority] += 1
    
    # Calculate average for each priority level
    waiting_time_by_priority = {
        priority: waiting_time / priority_counts[priority]
        for priority, waiting_time in priority_waiting_times.items()
    }
    
    metrics['waiting_time_by_priority'] = waiting_time_by_priority
    
    # Calculate priority preference ratio - how much high priority processes are favored
    if len(priority_waiting_times) > 1:
        min_priority = min(priority_waiting_times.keys())
        max_priority = max(priority_waiting_times.keys())
        low_pri_wait = waiting_time_by_priority[max_priority]
        high_pri_wait = waiting_time_by_priority[min_priority]
        
        # Fix the division by zero error by checking if high_pri_wait is zero
        if high_pri_wait > 0:
            metrics['priority_preference_ratio'] = low_pri_wait / high_pri_wait
        elif low_pri_wait > 0:
            # If high priority wait time is 0 but low priority wait time is positive,
            # the preference ratio is theoretically infinite
            metrics['priority_preference_ratio'] = float('inf')
        else:
            # If both are 0, set to 1.0 (neutral preference)
            metrics['priority_preference_ratio'] = 1.0
    else:
        metrics['priority_preference_ratio'] = 1.0
    
    return metrics

def calculate_rr_metrics(completed_processes, execution_sequence, time_quantum):
    """Calculate metrics for Round Robin scheduling"""
    metrics = calculate_basic_metrics(completed_processes)
    
    # Count context switches
    process_segments = {}
    for segment in execution_sequence:
        pid = segment['pid']
        if pid not in process_segments:
            process_segments[pid] = 0
        process_segments[pid] += 1
    
    # Total context switches = sum of segments - number of processes
    context_switches = sum(process_segments.values()) - len(process_segments)
    metrics['context_switches'] = context_switches
    
    # Calculate fairness - standard deviation of CPU time slices among processes
    if len(process_segments) > 1:
        avg_segments = sum(process_segments.values()) / len(process_segments)
        variance = sum((segments - avg_segments) ** 2 for segments in process_segments.values()) / len(process_segments)
        fairness = 1 / (1 + (variance ** 0.5))  # Lower deviation = higher fairness
        metrics['fairness'] = fairness
    else:
        metrics['fairness'] = 1.0
    
    # Quantum utilization - how effectively the time quantum is used
    quantum_utilization = 0
    quantum_count = 0
    
    for segment in execution_sequence:
        duration = segment['end'] - segment['start']
        if duration <= time_quantum:
            quantum_utilization += duration / time_quantum
            quantum_count += 1
    
    if quantum_count > 0:
        metrics['quantum_utilization'] = quantum_utilization / quantum_count
    else:
        metrics['quantum_utilization'] = 0
        
    return metrics

def calculate_priority_rr_metrics(completed_processes, execution_sequence, time_quantum):
    """Calculate metrics for Priority + Round Robin scheduling"""
    # Start with RR metrics
    metrics = calculate_rr_metrics(completed_processes, execution_sequence, time_quantum)
    
    # Add priority-based metrics
    priority_metrics = calculate_priority_metrics(completed_processes, execution_sequence)
    
    # Merge the priority-specific metrics
    metrics['waiting_time_by_priority'] = priority_metrics['waiting_time_by_priority']
    metrics['priority_preference_ratio'] = priority_metrics['priority_preference_ratio']
    
    return metrics

def compare_algorithms(results):
    """Compare results from different scheduling algorithms"""
    comparisons = {}
    
    # Find best algorithm for each metric
    # Common metrics found in all algorithms
    common_metrics = [
        'avg_turnaround_time', 
        'avg_waiting_time', 
        'avg_response_time',
        'throughput',
        'cpu_utilization'
    ]
    
    # Track metrics present in each algorithm's results
    all_metrics = set()
    for alg, alg_results in results.items():
        all_metrics.update(alg_results.keys())
    
    for metric in all_metrics:
        # Get algorithms that have this metric
        algs_with_metric = [alg for alg in results if metric in results[alg]]
        
        if not algs_with_metric:
            continue
            
        if metric in ['avg_turnaround_time', 'avg_waiting_time', 'avg_response_time', 'context_switches']:
            # Lower is better
            best_alg = min(
                [(alg, results[alg][metric]) for alg in algs_with_metric],
                key=lambda x: x[1]
            )[0]
        else:
            # Higher is better (throughput, cpu_utilization, fairness, etc.)
            best_alg = max(
                [(alg, results[alg][metric]) for alg in algs_with_metric],
                key=lambda x: x[1]
            )[0]
        
        comparisons[metric] = {
            'best_algorithm': best_alg,
            'values': {alg: results[alg][metric] for alg in algs_with_metric}
        }
    
    return comparisons
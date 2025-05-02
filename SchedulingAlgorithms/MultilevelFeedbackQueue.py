from SchedulingAlgorithms.Scheduler import Scheduler
import metrics
from collections import deque

class MFQScheduler(Scheduler):
    """
    Multilevel Feedback Queue Scheduler
    
    This advanced scheduler uses multiple queues with different priorities and time quantum values.
    Processes start in the highest priority queue and are moved to lower priority queues if they
    don't complete within their time quantum, implementing an adaptive approach that favors
    shorter processes while preventing starvation of longer processes.
    """
    
    def __init__(self, processes, num_queues=3, base_quantum=2):
        super().__init__(processes)
        self.num_queues = num_queues  # Number of priority queues
        self.base_quantum = base_quantum  # Base time quantum for highest priority queue
        self.queue_quanta = [self.base_quantum * (2**i) for i in range(num_queues)]
        self.context_switches = 0
    
    def schedule(self):
        """Execute Multilevel Feedback Queue scheduling algorithm"""
        # Sort processes by arrival time initially
        self.processes.sort(key=lambda p: p.arrival_time)
        
        # Create a copy of all processes
        remaining_processes = self.processes.copy()
        
        # Initialize multilevel queues
        queues = [deque() for _ in range(self.num_queues)]
        
        # Track process queue levels
        process_levels = {p.pid: 0 for p in self.processes}
        
        while remaining_processes or any(queues):
            # Check for new arrivals and add to highest priority queue
            newly_arrived = [p for p in remaining_processes if p.arrival_time <= self.current_time]
            for process in newly_arrived:
                queues[0].append(process)
                remaining_processes.remove(process)
            
            # If no process is in any queue and there are still processes to arrive
            if not any(queues) and remaining_processes:
                self.current_time = min(p.arrival_time for p in remaining_processes)
                continue
            
            # Process the highest non-empty queue first
            selected_queue = None
            for i, queue in enumerate(queues):
                if queue:
                    selected_queue = i
                    break
            
            if selected_queue is None:  # All queues are empty
                break
            
            current_process = queues[selected_queue].popleft()
            current_quantum = self.queue_quanta[selected_queue]
            
            # If this is the first time process is getting CPU, set start time
            if current_process.start_time is None:
                current_process.start_time = self.current_time
            
            # Calculate execution time (minimum of quantum and remaining time)
            execution_time = min(current_quantum, current_process.remaining_time)
            
            # Record execution in sequence
            self.execution_sequence.append({
                'pid': current_process.pid,
                'start': self.current_time,
                'end': self.current_time + execution_time,
                'queue': selected_queue
            })
            
            # Update current time and process's remaining time
            self.current_time += execution_time
            current_process.remaining_time -= execution_time
            
            # Check for newly arrived processes during this execution
            newly_arrived = [p for p in remaining_processes if p.arrival_time <= self.current_time]
            for process in newly_arrived:
                queues[0].append(process)
                remaining_processes.remove(process)
            
            # Check if process is completed
            if current_process.remaining_time == 0:
                current_process.completion_time = self.current_time
                self.completed_processes.append(current_process)
                self.context_switches += 1  # Count completion as context switch
            else:
                # If process consumed full quantum, demote to lower queue (unless already at lowest)
                if execution_time == current_quantum and selected_queue < self.num_queues - 1:
                    next_queue = selected_queue + 1
                    process_levels[current_process.pid] = next_queue
                else:
                    next_queue = selected_queue
                    
                # Put the process back in appropriate queue
                queues[next_queue].append(current_process)
                self.context_switches += 1  # Count preemption as context switch
        
        # Calculate metrics
        metrics_result = self.calculate_mfq_metrics()
        
        return metrics_result
    
    def calculate_mfq_metrics(self):
        """Calculate specific metrics for Multilevel Feedback Queue scheduler"""
        # Get basic metrics
        basic_metrics = metrics.calculate_basic_metrics(self.completed_processes)
        
        # Add MFQ-specific metrics
        mfq_metrics = {
            'context_switches': self.context_switches,
            'execution_sequence': self.execution_sequence
        }
        
        # Calculate average queue level (lower is better)
        queue_transitions = {}
        for segment in self.execution_sequence:
            pid = segment['pid']
            if pid not in queue_transitions:
                queue_transitions[pid] = []
            queue_transitions[pid].append(segment['queue'])
        
        avg_queue_level = sum(sum(levels) / len(levels) for levels in queue_transitions.values()) / len(queue_transitions)
        mfq_metrics['avg_queue_level'] = avg_queue_level
        
        # Count how many processes stayed in top queue (indicative of short processes)
        top_queue_processes = sum(1 for levels in queue_transitions.values() 
                                if all(level == 0 for level in levels))
        mfq_metrics['top_queue_ratio'] = top_queue_processes / len(queue_transitions)
        
        # Merge all metrics
        return {**basic_metrics, **mfq_metrics}
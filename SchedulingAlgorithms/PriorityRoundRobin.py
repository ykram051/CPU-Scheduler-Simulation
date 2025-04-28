from collections import deque
from . import Scheduler
import metrics

class PriorityRRScheduler(Scheduler):
    def __init__(self, processes, time_quantum=2):
        super().__init__(processes)
        self.time_quantum = time_quantum
        
    def schedule(self):
        # Sort processes by arrival time
        self.processes.sort(key=lambda p: p.arrival_time)
        
        # Create a copy of all processes
        remaining_processes = self.processes.copy()
        
        # Dictionary of priority queues (lower value = higher priority)
        priority_queues = {}
        
        while remaining_processes or any(priority_queues.values()):
            # Check for new arrivals and add to appropriate priority queue
            newly_arrived = [p for p in remaining_processes if p.arrival_time <= self.current_time]
            for process in newly_arrived:
                if process.priority not in priority_queues:
                    priority_queues[process.priority] = deque()
                priority_queues[process.priority].append(process)
                remaining_processes.remove(process)
            
            # If no process is in any ready queue and there are still processes to arrive
            if not any(priority_queues.values()) and remaining_processes:
                # Fast forward to the next arrival time
                self.current_time = min(p.arrival_time for p in remaining_processes)
                continue
            
            # If all queues are empty, we're done
            if not any(priority_queues.values()):
                break
            
            # Get process from highest priority non-empty queue
            priority_levels = sorted(priority_queues.keys())
            
            for priority in priority_levels:
                if priority_queues[priority]:
                    current_process = priority_queues[priority].popleft()
                    
                    # If this is the first time process is getting CPU, set start time
                    if current_process.start_time is None:
                        current_process.start_time = self.current_time
                    
                    # Calculate execution time (minimum of time quantum and remaining time)
                    execution_time = min(self.time_quantum, current_process.remaining_time)
                    
                    # Record execution in sequence
                    self.execution_sequence.append({
                        'pid': current_process.pid,
                        'start': self.current_time,
                        'end': self.current_time + execution_time
                    })
                    
                    # Update current time and process's remaining time
                    self.current_time += execution_time
                    current_process.remaining_time -= execution_time
                    
                    # Check for newly arrived processes during this execution
                    newly_arrived = [p for p in remaining_processes if p.arrival_time <= self.current_time]
                    for process in newly_arrived:
                        if process.priority not in priority_queues:
                            priority_queues[process.priority] = deque()
                        priority_queues[process.priority].append(process)
                        remaining_processes.remove(process)
                    
                    # Check if process is completed
                    if current_process.remaining_time == 0:
                        current_process.completion_time = self.current_time
                        self.completed_processes.append(current_process)
                    else:
                        # Put the process back in its priority queue
                        if current_process.priority not in priority_queues:
                            priority_queues[process.priority] = deque()
                        priority_queues[process.priority].append(current_process)
                    
                    # Break out of the priority levels loop to recheck arrivals and priorities
                    break
        
        # Calculate metrics using the metrics module
        return {
            **metrics.calculate_priority_rr_metrics(self.completed_processes, self.execution_sequence, self.time_quantum),
            'execution_sequence': self.execution_sequence
        }
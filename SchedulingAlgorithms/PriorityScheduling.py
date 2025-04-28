from SchedulingAlgorithms.Scheduler import Scheduler
import metrics

class PriorityScheduler(Scheduler):
    def __init__(self, processes):
        super().__init__(processes)
        
    def schedule(self):
        # Make a copy of processes to work with
        remaining_processes = self.processes.copy()
        
        # Sort processes by arrival time initially
        remaining_processes.sort(key=lambda p: p.arrival_time)
        
        # Continue until all processes are completed
        while remaining_processes:
            # Filter processes that have arrived by current time
            available_processes = [p for p in remaining_processes if p.arrival_time <= self.current_time]
            
            if not available_processes:
                # If no process is available, advance time to the next arrival
                self.current_time = min(p.arrival_time for p in remaining_processes)
                continue
            
            # Select process with highest priority (lowest priority number)
            selected_process = min(available_processes, key=lambda p: p.priority)
            
            # If this is the first time process is getting CPU, set start time
            if selected_process.start_time is None:
                selected_process.start_time = self.current_time
            
            # Execute the process (non-preemptive)
            self.execution_sequence.append({
                'pid': selected_process.pid,
                'start': self.current_time,
                'end': self.current_time + selected_process.remaining_time
            })
            
            # Update time
            self.current_time += selected_process.remaining_time
            
            # Mark process as completed
            selected_process.completion_time = self.current_time
            selected_process.remaining_time = 0
            self.completed_processes.append(selected_process)
            
            # Remove the process from remaining processes
            remaining_processes.remove(selected_process)
        
        # Calculate metrics using the metrics module
        return {
            **metrics.calculate_priority_metrics(self.completed_processes, self.execution_sequence),
            'execution_sequence': self.execution_sequence
        }
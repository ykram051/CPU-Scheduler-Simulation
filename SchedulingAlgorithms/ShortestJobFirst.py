from . import Scheduler
import metrics

class SJFScheduler(Scheduler):
    def __init__(self, processes):
        super().__init__(processes)
        
    def schedule(self):
        # Sort processes by arrival time initially
        self.processes.sort(key=lambda p: p.arrival_time)
        
        # Make a copy of processes to work with
        remaining_processes = self.processes.copy()
        
        # Continue until all processes are completed
        while remaining_processes:
            # Filter processes that have arrived by current time
            available_processes = [p for p in remaining_processes if p.arrival_time <= self.current_time]
            
            if not available_processes:
                # If no process is available, advance time to the next arrival
                self.current_time = min(p.arrival_time for p in remaining_processes)
                continue
                
            # Select process with shortest burst time (SJF)
            selected_process = min(available_processes, key=lambda p: p.burst_time)
            
            # If this is the first time process is getting CPU, set start time
            if selected_process.start_time is None:
                selected_process.start_time = self.current_time
            
            # Execute the process completely (SJF is non-preemptive in batch systems)
            self.execution_sequence.append({
                'pid': selected_process.pid,
                'start': self.current_time,
                'end': self.current_time + selected_process.remaining_time
            })
            
            # Update current time
            self.current_time += selected_process.remaining_time
            
            # Mark process as completed
            selected_process.completion_time = self.current_time
            selected_process.remaining_time = 0
            self.completed_processes.append(selected_process)
            
            # Remove the process from remaining processes
            remaining_processes.remove(selected_process)
        
        # Calculate metrics using the metrics module
        return {
            **metrics.calculate_sjf_metrics(self.completed_processes, self.execution_sequence),
            'execution_sequence': self.execution_sequence
        }
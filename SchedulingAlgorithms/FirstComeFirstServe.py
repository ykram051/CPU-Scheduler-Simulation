from SchedulingAlgorithms.Scheduler import Scheduler
import metrics

class FCFSScheduler(Scheduler):
    def __init__(self, processes):
        super().__init__(processes)
        
    def schedule(self):
        # Sort processes by arrival time
        self.processes.sort(key=lambda p: p.arrival_time)
        
        # Make a copy of processes to work with
        remaining_processes = self.processes.copy()
        
        # Continue until all processes are completed
        while remaining_processes:
            # Get the earliest arrived process that hasn't been processed yet
            # In FCFS, we always pick the process that arrived first
            current_process = remaining_processes[0]
            
            # If the process hasn't arrived yet, advance time to its arrival
            if current_process.arrival_time > self.current_time:
                self.current_time = current_process.arrival_time
                
            # If this is the first time process is getting CPU, set start time
            if current_process.start_time is None:
                current_process.start_time = self.current_time
            
            # Execute the process completely (FCFS is non-preemptive)
            self.execution_sequence.append({
                'pid': current_process.pid,
                'start': self.current_time,
                'end': self.current_time + current_process.remaining_time
            })
            
            # Update current time
            self.current_time += current_process.remaining_time
            
            # Mark process as completed
            current_process.completion_time = self.current_time
            current_process.remaining_time = 0
            self.completed_processes.append(current_process)
            
            # Remove the process from remaining processes
            remaining_processes.remove(current_process)
        
        # Calculate metrics using the metrics module
        return {
            **metrics.calculate_fcfs_metrics(self.completed_processes, self.execution_sequence),
            'execution_sequence': self.execution_sequence
        }
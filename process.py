class Process:
    """
    Class to represent a process in the CPU scheduling simulation.
    
    Attributes:
        pid (int): Process ID
        arrival_time (int): Time at which the process arrives in the ready queue
        burst_time (int): Total CPU time required by the process
        priority (int): Priority of the process (lower value means higher priority)
        remaining_time (int): Remaining CPU time required by the process
        completion_time (int): Time at which the process completes execution
        waiting_time (int): Total time spent by the process waiting in the ready queue
        turnaround_time (int): Total time taken from arrival to completion
        response_time (int): Time at which the process first gets the CPU
        start_time (int): Time at which the process starts execution
        started (bool): Flag indicating whether the process has started execution
    """
    
    def __init__(self, pid, arrival_time, burst_time, priority=0):
        
        self.pid = pid# Unique identifier for the process
        self.arrival_time = arrival_time # Time at which the process arrives in the ready queue
        self.burst_time = burst_time# Total CPU time required by the process
        self.priority = priority# Priority of the process (lower value means higher priority)
        
        # These will be calculated during scheduling
        self.remaining_time = burst_time
        self.completion_time = 0
        self.waiting_time = 0
        self.turnaround_time = 0
        self.response_time = None  # Time when process first gets CPU
        self.start_time = None    # Time when process starts execution
        self.started = False
    
    def __str__(self):
        return f"Process {self.pid} (Arrival: {self.arrival_time}, Burst: {self.burst_time}, Priority: {self.priority})"
    
    def reset(self):
        """
        Reset process state for a new scheduling simulation.
        """
        self.remaining_time = self.burst_time
        self.completion_time = 0
        self.waiting_time = 0
        self.turnaround_time = 0
        self.response_time = None
        self.start_time = None
        self.started = False
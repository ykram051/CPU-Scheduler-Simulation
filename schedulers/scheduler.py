class Scheduler:
    def __init__(self, processes):
        self.processes = processes.copy()  # Make a copy to avoid modifying original list
        self.current_time = 0
        self.execution_sequence = []  # List to store execution order for visualization
        self.ready_queue = []
        self.completed_processes = []

    def schedule(self):
        """Main scheduling method to be implemented by child classes"""
        raise NotImplementedError("Subclasses must implement schedule method")
import random
import process


def generate_random_processes(num_processes, arrival_min=0, arrival_max=20, 
                              burst_min=1, burst_max=20, priority_min=1, priority_max=10):
    """
    Generate a list of random processes.
    
    Args:
        num_processes (int): Number of processes to generate
        arrival_min/max (int): Range for random arrival times
        burst_min/max (int): Range for random burst times
        priority_min/max (int): Range for random priorities
        
    Returns:
        list: List of Process objects
    """
    processes = []
    
    for i in range(num_processes):
        pid = i + 1
        arrival_time = random.randint(arrival_min, arrival_max)
        burst_time = random.randint(burst_min, burst_max)
        priority = random.randint(priority_min, priority_max)
        
        processes.append(Process(pid, arrival_time, burst_time, priority))
    
    return processes

def save_processes_to_file(processes, filename="processes.csv"):
    """
    Save processes to a CSV file.
    
    Args:
        processes (list): List of Process objects
        filename (str): Name of the file to save to
    """
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['PID', 'Arrival Time', 'Burst Time', 'Priority'])  # Header
        
        for process in processes:
            writer.writerow([process.pid, process.arrival_time, process.burst_time, process.priority])
    
    print(f"Processes saved to {filename}")

def read_processes_from_file(filename="processes.csv"):
    """
    Read processes from a CSV file.
    
    Args:
        filename (str): Name of the file to read from
        
    Returns:
        list: List of Process objects
    """
    processes = []
    
    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            
            for row in reader:
                pid = int(row[0])
                arrival_time = int(row[1])
                burst_time = int(row[2])
                priority = int(row[3])
                
                processes.append(Process(pid, arrival_time, burst_time, priority))
        
        print(f"Processes loaded from {filename}")
    except FileNotFoundError:
        print(f"File {filename} not found.")
    except Exception as e:
        print(f"Error reading file: {e}")
    
    return processes
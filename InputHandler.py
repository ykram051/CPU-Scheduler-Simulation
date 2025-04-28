import csv
import random
import process
import csv
import json


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
        
        processes.append(process.Process(pid, arrival_time, burst_time, priority))
    
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
                
                processes.append(process.Process(pid, arrival_time, burst_time, priority))
        
        print(f"Processes loaded from {filename}")
    except FileNotFoundError:
        print(f"File {filename} not found.")
    except Exception as e:
        print(f"Error reading file: {e}")
    
    return processes

def save_processes_to_json(processes, filename="processes.json"):
    """
    Save processes to a JSON file.
    
    Args:
        processes (list): List of Process objects
        filename (str): Name of the file to save to
    """
    process_data = []
    
    for p in processes:
        process_data.append({
            "pid": p.pid,
            "arrival_time": p.arrival_time,
            "burst_time": p.burst_time,
            "priority": p.priority
        })
    
    with open(filename, 'w') as file:
        json.dump({"processes": process_data}, file, indent=4)
    
    print(f"Processes saved to {filename}")

def read_processes_from_json(filename="processes.json"):
    """
    Read processes from a JSON file.
    
    Args:
        filename (str): Name of the file to read from
        
    Returns:
        list: List of Process objects
    """
    processes = []
    
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
            
            for process_data in data.get("processes", []):
                pid = process_data.get("pid", 0)
                arrival_time = process_data.get("arrival_time", 0)
                burst_time = process_data.get("burst_time", 1)
                priority = process_data.get("priority", 1)
                
                processes.append(process.Process(pid, arrival_time, burst_time, priority))
        
        print(f"Processes loaded from {filename}")
    except FileNotFoundError:
        print(f"File {filename} not found.")
    except json.JSONDecodeError:
        print(f"Error parsing JSON in {filename}.")
    except Exception as e:
        print(f"Error reading file: {e}")
    
    return processes
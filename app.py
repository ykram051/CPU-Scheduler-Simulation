import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from process import Process
import metrics
import visualization
import InputHandler
from SchedulingAlgorithms.FirstComeFirstServe import FCFSScheduler
from SchedulingAlgorithms.ShortestJobFirst import SJFScheduler
from SchedulingAlgorithms.PriorityScheduling import PriorityScheduler
from SchedulingAlgorithms.RoundRobin import RoundRobinScheduler
from SchedulingAlgorithms.PriorityRoundRobin import PriorityRRScheduler

# Set page configuration
st.set_page_config(
    page_title="CPU Scheduler Simulator",
    page_icon="🖥️",
    layout="wide"
)

def main():
    # Title and description
    st.title("CPU Scheduler Simulation")
    st.markdown("""
    This application simulates different CPU scheduling algorithms and visualizes their performance.
    You can customize the processes and scheduling parameters below.
    """)
    
    # Sidebar for simulation configuration
    with st.sidebar:
        st.header("Simulation Configuration")
        
        # Option to randomly generate, manually input, or import processes
        input_method = st.radio("Process Input Method", 
                               ["Random Generation", "Manual Input", "Import from JSON File"])
        
        if input_method == "Random Generation":
            num_processes = st.slider("Number of Processes", min_value=3, max_value=20, value=5)
            min_arrival = st.slider("Minimum Arrival Time", min_value=0, max_value=20, value=0)
            max_arrival = st.slider("Maximum Arrival Time", min_value=min_arrival, max_value=50, value=10)
            min_burst = st.slider("Minimum Burst Time", min_value=1, max_value=20, value=1)
            max_burst = st.slider("Maximum Burst Time", min_value=min_burst, max_value=50, value=10)
            
            # Priority range if needed
            use_priority = st.checkbox("Include Priority Values", value=True)
            min_priority = 1
            max_priority = 10
            if use_priority:
                min_priority = st.slider("Minimum Priority", min_value=1, max_value=10, value=1)
                max_priority = st.slider("Maximum Priority", min_value=min_priority, max_value=20, value=10)
                st.info("Lower value means higher priority")
            
            # Generate random processes
            if st.button("Generate Processes"):
                processes = generate_random_processes(
                    num_processes, min_arrival, max_arrival, 
                    min_burst, max_burst, min_priority, max_priority
                )
                st.session_state.processes = processes
                st.success(f"Generated {num_processes} random processes!")
        
        elif input_method == "Manual Input":
            st.info("Enter process details manually")
            
            # Get number of processes
            num_manual_processes = st.number_input("Number of Processes", min_value=1, max_value=10, value=3)
            
            # Initialize processes list if not in session state
            if 'manual_processes' not in st.session_state:
                st.session_state.manual_processes = []
                
            # Process manual input in a form
            with st.form("process_input_form"):
                processes_data = []
                
                for i in range(num_manual_processes):
                    st.subheader(f"Process {i}")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        arrival = st.number_input(f"Arrival Time P{i}", min_value=0, value=i, key=f"arrival_{i}")
                    with col2:
                        burst = st.number_input(f"Burst Time P{i}", min_value=1, value=5, key=f"burst_{i}")
                    with col3:
                        priority = st.number_input(f"Priority P{i}", min_value=1, value=i+1, key=f"priority_{i}")
                    
                    processes_data.append({"pid": i, "arrival": arrival, "burst": burst, "priority": priority})
                
                submit_button = st.form_submit_button("Create Processes")
                
                if submit_button:
                    processes = create_processes_from_manual_input(processes_data)
                    st.session_state.processes = processes
                    st.success(f"Created {len(processes)} processes!")
        
        elif input_method == "Import from JSON File":
            st.info("Import processes from a JSON file")
            
            uploaded_file = st.file_uploader("Choose a JSON file", type="json")
            
            if uploaded_file is not None:
                # Save the uploaded file temporarily
                with open("temp_processes.json", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Read processes from the temp file
                processes = InputHandler.read_processes_from_json("temp_processes.json")
                
                if processes:
                    st.session_state.processes = processes
                    st.success(f"Imported {len(processes)} processes from JSON file!")
                else:
                    st.error("Failed to import processes from the file. Please check the file format.")
            
            st.markdown("### JSON Format Example")
            st.code("""
            {
                "processes": [
                    {
                        "pid": 1,
                        "arrival_time": 0,
                        "burst_time": 5,
                        "priority": 2
                    },
                    {
                        "pid": 2,
                        "arrival_time": 1,
                        "burst_time": 3,
                        "priority": 1
                    }
                ]
            }
            """, language="json")
        
        # Scheduler selection
        st.header("Scheduler Selection")
        scheduler_type = st.selectbox(
            "Select Scheduling Algorithm",
            ["First-Come, First-Served (FCFS)", 
             "Shortest Job First (SJF)",
             "Priority Scheduling",
             "Round Robin (RR)",
             "Priority Round Robin"]
        )
        
        # Round Robin quantum time if needed
        time_quantum = 2
        if "Round Robin" in scheduler_type:
            time_quantum = st.slider("Time Quantum", min_value=1, max_value=10, value=2)
        
        # Run simulation button
        run_simulation = st.button("Run Simulation")
        
        # Compare algorithms
        compare_algorithms = st.checkbox("Compare All Algorithms", value=False)
    
        # Add export to JSON option if processes are loaded
        if "processes" in st.session_state and st.session_state.processes:
            st.header("Export Options")
            if st.button("Export Processes to JSON"):
                # Create a downloadable JSON file
                process_data = []
                for p in st.session_state.processes:
                    process_data.append({
                        "pid": p.pid,
                        "arrival_time": p.arrival_time,
                        "burst_time": p.burst_time,
                        "priority": p.priority
                    })
                
                json_str = json.dumps({"processes": process_data}, indent=4)
                
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="processes.json",
                    mime="application/json"
                )
    
    # Main content area
    if "processes" in st.session_state:
        # Display process table
        st.header("Processes")
        process_df = processes_to_dataframe(st.session_state.processes)
        st.dataframe(process_df, use_container_width=True)
        
        # Run simulation if button was clicked
        if run_simulation:
            if compare_algorithms:
                run_comparison(st.session_state.processes, time_quantum)
            else:
                run_single_simulation(st.session_state.processes, scheduler_type, time_quantum)
    else:
        # Display placeholder content
        st.info("Configure simulation parameters in the sidebar and generate processes to start.")

def generate_random_processes(num, min_arrival, max_arrival, min_burst, max_burst, min_priority, max_priority):
    """Generate random processes with the given parameters"""
    processes = []
    for i in range(num):
        pid = i
        arrival_time = np.random.randint(min_arrival, max_arrival + 1)
        burst_time = np.random.randint(min_burst, max_burst + 1)
        priority = np.random.randint(min_priority, max_priority + 1)
        
        process = Process(pid, arrival_time, burst_time, priority)
        processes.append(process)
    
    return processes

def create_processes_from_manual_input(processes_data):
    """Create process objects from manually entered data"""
    processes = []
    for p_data in processes_data:
        process = Process(
            p_data["pid"], 
            p_data["arrival"], 
            p_data["burst"], 
            p_data["priority"]
        )
        processes.append(process)
    
    return processes

def processes_to_dataframe(processes):
    """Convert list of Process objects to a pandas DataFrame"""
    data = []
    for p in processes:
        data.append({
            "Process ID": p.pid,
            "Arrival Time": p.arrival_time,
            "Burst Time": p.burst_time,
            "Priority": p.priority,
            "Remaining Time": p.remaining_time
        })
    
    return pd.DataFrame(data)

def run_single_simulation(processes, scheduler_type, time_quantum):
    """Run simulation for a single scheduling algorithm"""
    st.header(f"Simulation Results: {scheduler_type}")
    
    # Create a copy of processes to avoid modifying the original
    process_copies = [Process(p.pid, p.arrival_time, p.burst_time, p.priority) for p in processes]
    
    # Select scheduler based on the type
    scheduler = None
    if scheduler_type == "First-Come, First-Served (FCFS)":
        scheduler = FCFSScheduler(process_copies)
    elif scheduler_type == "Shortest Job First (SJF)":
        scheduler = SJFScheduler(process_copies)
    elif scheduler_type == "Priority Scheduling":
        scheduler = PriorityScheduler(process_copies)
    elif scheduler_type == "Round Robin (RR)":
        scheduler = RoundRobinScheduler(process_copies, time_quantum)
    elif scheduler_type == "Priority Round Robin":
        scheduler = PriorityRRScheduler(process_copies, time_quantum)
    
    # Run the scheduler
    results = scheduler.schedule()
    
    # Display results
    display_simulation_results(results, scheduler.completed_processes)

def run_comparison(processes, time_quantum):
    """Run simulation for all scheduling algorithms and compare results"""
    st.header("Algorithm Comparison")
    
    # Define schedulers to compare
    schedulers = {
        "FCFS": FCFSScheduler,
        "SJF": SJFScheduler,
        "Priority": PriorityScheduler,
        "Round Robin": lambda p: RoundRobinScheduler(p, time_quantum),
        "Priority RR": lambda p: PriorityRRScheduler(p, time_quantum)
    }
    
    # Run all schedulers
    results = {}
    detailed_results = {}
    completed_processes = {}
    
    for name, scheduler_class in schedulers.items():
        # Create fresh copies of processes for each scheduler
        process_copies = [Process(p.pid, p.arrival_time, p.burst_time, p.priority) for p in processes]
        
        # Create and run scheduler
        scheduler = scheduler_class(process_copies)
        result = scheduler.schedule()
        
        # Save results
        results[name] = result
        completed_processes[name] = scheduler.completed_processes
    
    # Display comparison tabs
    tabs = st.tabs(["Performance Metrics", "Gantt Charts", "Turnaround Times", "Waiting Times"])
    
    with tabs[0]:
        display_metrics_comparison(results)
    
    with tabs[1]:
        display_gantt_comparison(results)
    
    with tabs[2]:
        display_turnaround_comparison(completed_processes)
    
    with tabs[3]:
        display_waiting_comparison(completed_processes)

def display_metrics_comparison(results):
    """Display a comparison table of key metrics"""
    # Extract metrics
    metrics_data = []
    for algorithm, result in results.items():
        metrics_data.append({
            "Algorithm": algorithm,
            "Avg. Turnaround Time": result.get("avg_turnaround_time", 0),
            "Avg. Waiting Time": result.get("avg_waiting_time", 0),
            "Avg. Response Time": result.get("avg_response_time", 0),
            "Throughput": result.get("throughput", 0),
            "CPU Utilization (%)": result.get("cpu_utilization", 0)
        })
    
    # Create DataFrame and display
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)
    
    # Create comparison charts
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Algorithms on x-axis
    algorithms = [data["Algorithm"] for data in metrics_data]
    
    # Turnaround time
    turnaround_times = [data["Avg. Turnaround Time"] for data in metrics_data]
    axes[0].bar(algorithms, turnaround_times)
    axes[0].set_title("Average Turnaround Time")
    axes[0].set_ylabel("Time")
    axes[0].tick_params(axis='x', rotation=45)
    
    # Waiting time
    waiting_times = [data["Avg. Waiting Time"] for data in metrics_data]
    axes[1].bar(algorithms, waiting_times)
    axes[1].set_title("Average Waiting Time")
    axes[1].set_ylabel("Time")
    axes[1].tick_params(axis='x', rotation=45)
    
    # CPU Utilization
    cpu_util = [data["CPU Utilization (%)"] for data in metrics_data]
    axes[2].bar(algorithms, cpu_util)
    axes[2].set_title("CPU Utilization (%)")
    axes[2].set_ylabel("Percentage")
    axes[2].set_ylim(0, 100)
    axes[2].tick_params(axis='x', rotation=45)
    
    fig.tight_layout()
    st.pyplot(fig)

def display_gantt_comparison(results):
    """Display Gantt charts for all algorithms side by side"""
    # Create subplots
    algorithms = list(results.keys())
    num_algs = len(algorithms)
    
    # Calculate subplot layout (trying to make it 2x3 or similar)
    if num_algs <= 3:
        rows, cols = 1, num_algs
    else:
        rows = 2
        cols = (num_algs + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]  # Make it iterable
    elif rows == 1 or cols == 1:
        axes = axes.flat
    
    # Plot each algorithm's Gantt chart
    for i, algorithm in enumerate(algorithms):
        ax = axes[i] if rows == 1 or cols == 1 else axes[i // cols, i % cols]
        execution_sequence = results[algorithm]["execution_sequence"]
        
        # Create colors for processes
        unique_pids = set(segment["pid"] for segment in execution_sequence)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_pids)))
        pid_to_color = {pid: colors[i] for i, pid in enumerate(unique_pids)}
        
        # Plot bars for each process execution
        for segment in execution_sequence:
            pid = segment["pid"]
            start = segment["start"]
            end = segment["end"]
            ax.barh(0, end - start, left=start, height=0.5, color=pid_to_color[pid], 
                    edgecolor="black", alpha=0.8)
            ax.text(start + (end - start) / 2, 0, f"P{pid}", ha="center", va="center", fontsize=8)
        
        # Set title and labels
        ax.set_title(f"{algorithm} Gantt Chart")
        ax.set_xlabel("Time")
        ax.set_yticks([])
        ax.grid(axis="x", linestyle="--", alpha=0.7)
    
    # Hide unused subplots if any
    for i in range(len(algorithms), rows * cols):
        if rows == 1 or cols == 1:
            fig.delaxes(axes[i])
        else:
            fig.delaxes(axes[i // cols, i % cols])
    
    fig.tight_layout()
    st.pyplot(fig)

def display_turnaround_comparison(completed_processes_dict):
    """Display turnaround time comparison by process for all algorithms"""
    # Create data for comparison
    data = []
    for algorithm, processes in completed_processes_dict.items():
        for process in processes:
            data.append({
                "Algorithm": algorithm,
                "Process ID": f"P{process.pid}",
                "Turnaround Time": process.completion_time - process.arrival_time
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Pivot the data for plotting
    pivot_df = df.pivot(index="Process ID", columns="Algorithm", values="Turnaround Time")
    pivot_df.plot(kind="bar", ax=ax)
    
    ax.set_title("Turnaround Time Comparison by Process")
    ax.set_ylabel("Turnaround Time")
    ax.legend(title="Algorithm")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    fig.tight_layout()
    st.pyplot(fig)
    
    # Show raw data
    st.dataframe(pivot_df, use_container_width=True)

def display_waiting_comparison(completed_processes_dict):
    """Display waiting time comparison by process for all algorithms"""
    # Create data for comparison
    data = []
    for algorithm, processes in completed_processes_dict.items():
        for process in processes:
            turnaround_time = process.completion_time - process.arrival_time
            waiting_time = turnaround_time - process.burst_time
            data.append({
                "Algorithm": algorithm,
                "Process ID": f"P{process.pid}",
                "Waiting Time": waiting_time
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Pivot the data for plotting
    pivot_df = df.pivot(index="Process ID", columns="Algorithm", values="Waiting Time")
    pivot_df.plot(kind="bar", ax=ax)
    
    ax.set_title("Waiting Time Comparison by Process")
    ax.set_ylabel("Waiting Time")
    ax.legend(title="Algorithm")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    fig.tight_layout()
    st.pyplot(fig)
    
    # Show raw data
    st.dataframe(pivot_df, use_container_width=True)

def display_simulation_results(results, completed_processes):
    """Display detailed results for a single simulation"""
    # Create tabs for different visualizations
    tabs = st.tabs(["Gantt Chart", "Metrics", "Process Timeline", "Process Details"])
    
    with tabs[0]:
        st.subheader("Gantt Chart")
        display_gantt_chart(results["execution_sequence"])
    
    with tabs[1]:
        st.subheader("Performance Metrics")
        # Extract and display metrics
        metrics_to_display = {
            "Average Turnaround Time": results.get("avg_turnaround_time", 0),
            "Average Waiting Time": results.get("avg_waiting_time", 0),
            "Average Response Time": results.get("avg_response_time", 0),
            "Throughput (processes/time unit)": results.get("throughput", 0),
            "CPU Utilization (%)": results.get("cpu_utilization", 0)
        }
        
        # Display metrics as columns
        cols = st.columns(len(metrics_to_display))
        for i, (metric_name, value) in enumerate(metrics_to_display.items()):
            cols[i].metric(metric_name, f"{value:.2f}")
        
        # Display other algorithm-specific metrics if available
        st.subheader("Additional Metrics")
        other_metrics = {}
        for key, value in results.items():
            if key not in ["avg_turnaround_time", "avg_waiting_time", "avg_response_time", 
                          "throughput", "cpu_utilization", "execution_sequence"]:
                other_metrics[key] = value
        
        if other_metrics:
            st.json(other_metrics)
        else:
            st.info("No additional metrics available for this algorithm.")
    
    with tabs[2]:
        st.subheader("Process Timeline")
        display_process_timeline(completed_processes)
    
    with tabs[3]:
        st.subheader("Process Details")
        display_process_details(completed_processes)

def display_gantt_chart(execution_sequence):
    """Display a Gantt chart for the execution sequence"""
    if not execution_sequence:
        st.warning("No execution sequence data available.")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Create colors for processes
    unique_pids = set(segment["pid"] for segment in execution_sequence)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_pids)))
    pid_to_color = {pid: colors[i] for i, pid in enumerate(unique_pids)}
    
    # Plot bars for each process execution
    y_pos = 0  # Only one row for the Gantt chart
    for segment in execution_sequence:
        pid = segment["pid"]
        start = segment["start"]
        end = segment["end"]
        ax.barh(y_pos, end - start, left=start, height=0.5, color=pid_to_color[pid], 
                edgecolor="black", alpha=0.8)
        ax.text(start + (end - start) / 2, y_pos, f"P{pid}", ha="center", va="center")
    
    # Set labels and grid
    ax.set_xlabel("Time")
    ax.set_yticks([])
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    
    # Add legend for processes
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=pid_to_color[pid], edgecolor="black", alpha=0.8, 
                           label=f"Process {pid}") for pid in sorted(unique_pids)]
    ax.legend(handles=legend_elements, loc="upper right")
    
    fig.tight_layout()
    st.pyplot(fig)

def display_process_timeline(completed_processes):
    """Display a timeline of each process showing its states"""
    if not completed_processes:
        st.warning("No completed processes data available.")
        return
    
    # Sort processes by arrival time
    processes = sorted(completed_processes, key=lambda p: p.arrival_time)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, len(processes) * 0.8))
    
    # Plot timeline for each process
    for i, process in enumerate(processes):
        # Arrival time
        ax.plot(process.arrival_time, i, marker="o", markersize=8, color="green", label="Arrival" if i == 0 else "")
        
        # Start time (first time on CPU)
        ax.plot(process.start_time, i, marker="^", markersize=8, color="blue", label="Start" if i == 0 else "")
        
        # Completion time
        ax.plot(process.completion_time, i, marker="s", markersize=8, color="red", label="Completion" if i == 0 else "")
        
        # Waiting time (from arrival to start)
        ax.plot([process.arrival_time, process.start_time], [i, i], linestyle="--", color="gray", alpha=0.5)
        
        # Process execution time
        ax.plot([process.start_time, process.completion_time], [i, i], linewidth=2, color="blue", alpha=0.7)
        
        # Annotate times
        ax.text(process.arrival_time, i + 0.1, f"{process.arrival_time}", fontsize=8)
        ax.text(process.start_time, i + 0.1, f"{process.start_time}", fontsize=8)
        ax.text(process.completion_time, i + 0.1, f"{process.completion_time}", fontsize=8)
    
    # Set labels and ticks
    ax.set_xlabel("Time")
    ax.set_yticks(range(len(processes)))
    ax.set_yticklabels([f"Process {p.pid}" for p in processes])
    ax.grid(axis="x", linestyle="--", alpha=0.7)
    
    # Add legend
    ax.legend(loc="upper right")
    
    fig.tight_layout()
    st.pyplot(fig)

def display_process_details(completed_processes):
    """Display detailed information for each completed process"""
    if not completed_processes:
        st.warning("No completed processes data available.")
        return
    
    # Create DataFrame with process details
    data = []
    for p in completed_processes:
        turnaround_time = p.completion_time - p.arrival_time
        waiting_time = turnaround_time - p.burst_time
        response_time = p.start_time - p.arrival_time
        
        data.append({
            "Process ID": p.pid,
            "Arrival Time": p.arrival_time,
            "Burst Time": p.burst_time,
            "Priority": p.priority,
            "Start Time": p.start_time,
            "Completion Time": p.completion_time,
            "Turnaround Time": turnaround_time,
            "Waiting Time": waiting_time,
            "Response Time": response_time
        })
    
    # Display as DataFrame
    df = pd.DataFrame(data)
    st.dataframe(df.style.highlight_max(axis=0, subset=["Turnaround Time", "Waiting Time", "Response Time"], color="yellow")
                       .highlight_min(axis=0, subset=["Turnaround Time", "Waiting Time", "Response Time"], color="lightgreen"),
                 use_container_width=True)
    
    # Calculate and display averages
    st.subheader("Average Values")
    avg_data = {
        "Metric": ["Turnaround Time", "Waiting Time", "Response Time"],
        "Average": [
            df["Turnaround Time"].mean(),
            df["Waiting Time"].mean(),
            df["Response Time"].mean()
        ]
    }
    avg_df = pd.DataFrame(avg_data)
    st.dataframe(avg_df, use_container_width=True)

if __name__ == "__main__":
    main()
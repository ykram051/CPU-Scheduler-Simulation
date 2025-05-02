import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import json
import datetime
import time
from PIL import Image

from process import Process
from SchedulingAlgorithms.FirstComeFirstServe import FCFSScheduler
from SchedulingAlgorithms.ShortestJobFirst import SJFScheduler
from SchedulingAlgorithms.PriorityScheduling import PriorityScheduler
from SchedulingAlgorithms.RoundRobin import RoundRobinScheduler
from SchedulingAlgorithms.PriorityRoundRobin import PriorityRRScheduler
from SchedulingAlgorithms.MultilevelFeedbackQueue import MFQScheduler

import visualization
import advanced_visualizations
from config import get_config, update_config, SIMULATION_CONFIG

# Page configuration
st.set_page_config(
    page_title="CPU Scheduler Simulator",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper Functions
def reset_processes(processes):
    """Reset all processes to initial state"""
    for p in processes:
        p.reset()
    return processes

def generate_random_processes(count):
    """Generate random processes for simulation"""
    random_config = get_config("random_process")
    processes = []
    
    for i in range(1, count + 1):
        arrival_time = np.random.randint(
            random_config["min_arrival_time"], 
            random_config["max_arrival_time"]
        )
        burst_time = np.random.randint(
            random_config["min_burst_time"], 
            random_config["max_burst_time"]
        )
        priority = np.random.randint(
            random_config["min_priority"], 
            random_config["max_priority"]
        )
        processes.append(Process(i, arrival_time, burst_time, priority))
        
    return processes

def save_processes_to_file(processes, filename):
    """Save processes to a JSON file"""
    with open(filename, 'w') as f:
        json.dump([{
            'pid': p.pid,
            'arrival_time': p.arrival_time,
            'burst_time': p.burst_time,
            'priority': p.priority
        } for p in processes], f, indent=4)

def load_processes_from_file(filename):
    """Load processes from a JSON file"""
    with open(filename, 'r') as f:
        data = json.load(f)
        return [Process(
            p['pid'], 
            p['arrival_time'], 
            p['burst_time'], 
            p['priority']
        ) for p in data]

def plot_to_image(fig):
    """Convert a matplotlib figure to an image for display in Streamlit"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    return Image.open(buf)

# Initialize session state if not already done
if 'processes' not in st.session_state:
    st.session_state.processes = generate_random_processes(5)
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'selected_algorithms' not in st.session_state:
    st.session_state.selected_algorithms = ["FCFS", "SJF", "Priority", "RR", "Priority RR", "MFQ"]
if 'show_advanced' not in st.session_state:
    st.session_state.show_advanced = get_config("ui", "show_advanced_options")
if 'theme' not in st.session_state:
    st.session_state.theme = get_config("ui", "theme")

# App Title and Description
st.title("🖥️ CPU Scheduler Simulation")
st.markdown("""
This application simulates different CPU scheduling algorithms. You can:
- Generate random processes or enter your own processes
- Run various scheduling algorithms (FCFS, SJF, Priority, etc.)
- Visualize results with graphs and compare performance
""")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Process Management
    st.subheader("Process Management")
    process_option = st.radio(
        "Process source",
        ["Use example", "Generate randomly", "Define manually", "Load from file"]
    )
    
    if process_option == "Use example":
        example = st.selectbox(
            "Choose example",
            ["Sequential processes", "Overlapping processes", "Same arrival time", 
             "Edge cases (very short/long)", "Priority inversion"]
        )
        
        if example == "Sequential processes":
            st.session_state.processes = [
                Process(1, 0, 5, 3),
                Process(2, 5, 3, 1),
                Process(3, 8, 2, 2)
            ]
        elif example == "Overlapping processes":
            st.session_state.processes = [
                Process(1, 0, 7, 3),
                Process(2, 2, 4, 1),
                Process(3, 4, 1, 2),
                Process(4, 6, 3, 4)
            ]
        elif example == "Same arrival time":
            st.session_state.processes = [
                Process(1, 0, 6, 3),
                Process(2, 0, 3, 1),
                Process(3, 0, 8, 2)
            ]
        elif example == "Edge cases (very short/long)":
            st.session_state.processes = [
                Process(1, 0, 1, 5),
                Process(2, 1, 20, 2),
                Process(3, 2, 2, 1),
                Process(4, 3, 15, 10),
                Process(5, 5, 5, 3)
            ]
        elif example == "Priority inversion":
            st.session_state.processes = [
                Process(1, 0, 5, 3),
                Process(2, 2, 10, 1),
                Process(3, 1, 3, 10)
            ]
            
    elif process_option == "Generate randomly":
        num_processes = st.slider(
            "Number of processes",
            min_value=2,
            max_value=20,
            value=5,
            step=1
        )
        
        if st.button("🎲 Generate processes"):
            st.session_state.processes = generate_random_processes(num_processes)
            st.success(f"{num_processes} processes generated successfully!")
            
    elif process_option == "Define manually":
        num_processes = st.slider(
            "Number of processes",
            min_value=1,
            max_value=10,
            value=3,
            step=1
        )
        
        processes = []
        for i in range(1, num_processes + 1):
            st.subheader(f"Process {i}")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                arrival = st.number_input(f"Arrival time P{i}", min_value=0, value=i-1, step=1, key=f"arrival_{i}")
            with col2:
                burst = st.number_input(f"Burst time P{i}", min_value=1, value=5, step=1, key=f"burst_{i}")
            with col3:
                priority = st.number_input(f"Priority P{i}", min_value=1, value=i, step=1, key=f"priority_{i}")
                
            processes.append(Process(i, arrival, burst, priority))
        
        st.session_state.processes = processes
        
    elif process_option == "Load from file":
        uploaded_file = st.file_uploader("Load process JSON file", type=["json"])
        if uploaded_file:
            try:
                # Save uploaded file temporarily and load processes
                with open("temp_processes.json", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.processes = load_processes_from_file("temp_processes.json")
                st.success(f"{len(st.session_state.processes)} processes loaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    # Algorithm Selection
    st.subheader("Algorithms")
    algorithms = {
        "FCFS": "First Come First Serve",
        "SJF": "Shortest Job First", 
        "Priority": "Priority Scheduling",
        "RR": "Round Robin",
        "Priority RR": "Priority Round Robin",
        "MFQ": "Multilevel Feedback Queue"
    }
    
    st.session_state.selected_algorithms = []
    for short_name, full_name in algorithms.items():
        if st.checkbox(full_name, value=True, key=f"alg_{short_name}"):
            st.session_state.selected_algorithms.append(short_name)
    
    # Algorithm Parameters
    st.subheader("Algorithm Parameters")
    rr_quantum = st.slider(
        "Round Robin quantum",
        min_value=1,
        max_value=10,
        value=get_config("round_robin", "default_quantum"),
        step=1
    )
    
    mfq_queues = st.slider(
        "MFQ number of queues",
        min_value=2,
        max_value=5,
        value=get_config("multilevel_feedback_queue", "num_queues"),
        step=1
    )
    
    mfq_quantum = st.slider(
        "MFQ base quantum",
        min_value=1,
        max_value=5,
        value=get_config("multilevel_feedback_queue", "base_quantum"),
        step=1
    )
    
    # Advanced options toggle
    st.subheader("Advanced Options")
    show_advanced = st.checkbox("Show advanced options", value=st.session_state.show_advanced)
    if show_advanced != st.session_state.show_advanced:
        st.session_state.show_advanced = show_advanced
        update_config("ui", "show_advanced_options", show_advanced)
    
    if show_advanced:
        visualization_type = st.radio(
            "Visualization type",
            ["Standard", "Advanced", "Full Dashboard"],
            index=1
        )
        
        theme = st.selectbox(
            "UI theme",
            ["light", "dark"],
            index=0 if st.session_state.theme == "light" else 1
        )
        if theme != st.session_state.theme:
            st.session_state.theme = theme
            update_config("ui", "theme", theme)
    else:
        visualization_type = "Standard"

# Main area - Process table
st.header("Process List")
process_data = [
    {"ID": p.pid, "Arrival Time": p.arrival_time, "Burst Time": p.burst_time, "Priority": p.priority} 
    for p in st.session_state.processes
]
df_processes = pd.DataFrame(process_data)
st.table(df_processes)

# Save and load processes
col1, col2 = st.columns(2)
with col1:
    save_filename = st.text_input("Filename to save", value="processes.json")
    if st.button("💾 Save processes"):
        try:
            save_processes_to_file(st.session_state.processes, save_filename)
            st.success(f"Processes saved to {save_filename}")
        except Exception as e:
            st.error(f"Error saving: {str(e)}")

# Run simulation button
if st.button("▶️ Run simulation", type="primary"):
    st.session_state.results = {}
    
    with st.spinner("Running simulation..."):
        # Mapping from short names to scheduler classes
        scheduler_map = {
            "FCFS": FCFSScheduler,
            "SJF": SJFScheduler,
            "Priority": PriorityScheduler,
            "RR": lambda p: RoundRobinScheduler(p, rr_quantum),
            "Priority RR": lambda p: PriorityRRScheduler(p, rr_quantum),
            "MFQ": lambda p: MFQScheduler(p, mfq_queues, mfq_quantum)
        }
        
        # Run selected algorithms
        for alg_name in st.session_state.selected_algorithms:
            if alg_name in scheduler_map:
                # Reset processes and run algorithm
                scheduler = scheduler_map[alg_name](reset_processes(st.session_state.processes.copy()))
                result = scheduler.schedule()
                
                # Store results using full algorithm name
                full_name = algorithms[alg_name]
                st.session_state.results[full_name] = result
    
    st.success("Simulation complete!")

# Results section
if st.session_state.results:
    st.header("Simulation Results")
    
    # Metrics table
    st.subheader("Metrics")
    
    # Prepare metrics DataFrame
    metrics_data = []
    for alg_name, result in st.session_state.results.items():
        metrics_dict = {
            "Algorithm": alg_name,
            "Average Waiting Time": round(result["avg_waiting_time"], 2),
            "Average Turnaround Time": round(result["avg_turnaround_time"], 2),
            "Average Response Time": round(result["avg_response_time"], 2),
            "Throughput": round(result.get("throughput", 0), 2),
            "CPU Utilization (%)": round(result.get("cpu_utilization", 0), 2)
        }
        
        # Add algorithm-specific metrics
        if "context_switches" in result:
            metrics_dict["Context Switches"] = result["context_switches"]
        if "fairness" in result:
            metrics_dict["Fairness"] = round(result["fairness"], 2)
        if "avg_queue_level" in result:
            metrics_dict["Average Queue Level"] = round(result["avg_queue_level"], 2)
        if "top_queue_ratio" in result:
            metrics_dict["Top Queue Ratio"] = round(result["top_queue_ratio"], 2)
            
        metrics_data.append(metrics_dict)
    
    # Display table
    metrics_df = pd.DataFrame(metrics_data)
    st.table(metrics_df)
    
    # Visualizations based on selected type
    st.subheader("Visualizations")
    
    if visualization_type in ["Standard", "Advanced"]:
        # Tab options for different visualization types
        tab1, tab2, tab3 = st.tabs(["Gantt Charts", "Process Timelines", "Comparisons"])
        
        # Gantt charts
        with tab1:
            for alg_name, result in st.session_state.results.items():
                if "execution_sequence" in result:
                    st.write(f"### {alg_name}")
                    
                    if visualization_type == "Advanced":
                        fig = advanced_visualizations.create_process_lifecycle_visualization(
                            result["execution_sequence"], 
                            st.session_state.processes, 
                            alg_name,
                            None  # Don't save to file
                        )
                        if fig:  # Vérifier que la figure existe
                            st.pyplot(fig)
                    else:
                        fig = visualization.create_gantt_chart(
                            result["execution_sequence"], 
                            alg_name,
                            None  # Don't save to file
                        )
                        if fig:  # Vérifier que la figure existe
                            st.pyplot(fig)
        
        # Process timelines
        with tab2:
            for alg_name, result in st.session_state.results.items():
                if "execution_sequence" in result and "completed_processes" in result:
                    st.write(f"### {alg_name}")
                    
                    fig = visualization.create_process_timeline(
                        result["completed_processes"],
                        alg_name,
                        None  # Don't save to file
                    )
                    if fig:  # Vérifier que la figure existe
                        st.pyplot(fig)
        
        # Comparisons
        with tab3:
            st.write("### Algorithm Comparison")
            
            if visualization_type == "Advanced" and len(st.session_state.results) > 1:
                # Create multiple comparison visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("#### Metrics Heatmap")
                    fig = advanced_visualizations.create_heatmap(
                        st.session_state.results, 
                        ['avg_waiting_time', 'avg_turnaround_time', 'avg_response_time', 'throughput', 'cpu_utilization'], 
                        "Performance Comparison", 
                        None
                    )
                    if fig:
                        st.pyplot(fig)
                
                with col2:
                    st.write("#### Metrics Radar Chart")
                    all_metrics = set()
                    for alg, metrics_dict in st.session_state.results.items():
                        all_metrics.update([m for m in metrics_dict.keys() if isinstance(metrics_dict[m], (int, float)) and 
                                          m not in ('execution_sequence', 'completed_processes')])
                    fig = advanced_visualizations.create_radar_chart(
                        st.session_state.results, 
                        list(all_metrics), 
                        "Metrics Comparison", 
                        None
                    )
                    if fig:
                        st.pyplot(fig)
            else:
                fig = visualization.create_metrics_comparison_chart(
                    {alg: {k: v for k, v in result.items() if k not in ['execution_sequence', 'completed_processes'] 
                          and not isinstance(v, dict)} for alg, result in st.session_state.results.items()}, 
                    None
                )
                if fig:
                    st.pyplot(fig)
    
    elif visualization_type == "Full Dashboard":
        st.write("### Full Dashboard")
        
        # Generate dashboard in a temporary directory and display components
        with st.spinner("Generating dashboard..."):
            # Create temporary directory for dashboard
            dashboard_dir = os.path.join("temp_dashboard", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(dashboard_dir, exist_ok=True)
            
            # Generate dashboard files - les fonctions de visualisation retournent les chemins des fichiers créés
            dashboard = advanced_visualizations.create_comparative_dashboard(
                st.session_state.results, st.session_state.processes, "CPU Scheduler Comparison", dashboard_dir
            )
            
            # Display components from saved files
            if "heatmap" in dashboard and os.path.exists(dashboard["heatmap"]):
                st.write("#### Common Metrics Heatmap")
                st.image(dashboard["heatmap"])
            
            if "radar" in dashboard and os.path.exists(dashboard["radar"]):
                st.write("#### All Metrics Radar Chart")
                st.image(dashboard["radar"])
            
            if "lifecycles" in dashboard and dashboard["lifecycles"]:
                st.write("#### Process Lifecycle Visualizations")
                # Display lifecycles in columns
                cols = st.columns(2)
                for i, img_path in enumerate(dashboard["lifecycles"]):
                    if os.path.exists(img_path):
                        with cols[i % 2]:
                            st.image(img_path)
    
    # Add option to download a PDF report
    st.subheader("Reports")
    if st.button("📊 Generate PDF Report"):
        with st.spinner("Generating report..."):
            # In a real implementation, this would generate a PDF report
            # Here we'll simulate a delay and provide a download link
            time.sleep(2)
            st.success("Report generated successfully!")
            # In a real app, this would be a link to download the actual report
            st.download_button(
                label="📥 Download PDF Report", 
                data=b"PDF Report Simulation", 
                file_name="cpu_scheduler_report.pdf"
            )

# Footer
st.markdown("---")
st.markdown("### About")
st.markdown("""
This application allows simulating and analyzing different CPU scheduling algorithms, including:
- **First Come First Serve (FCFS)**
- **Shortest Job First (SJF)**
- **Priority Scheduling**
- **Round Robin**
- **Priority Round Robin**
- **Multilevel Feedback Queue (MFQ)**

Developed as part of an academic project in computational theory.
""")

if st.session_state.show_advanced:
    st.markdown("### Advanced Settings")
    with st.expander("Current Configuration"):
        st.json(SIMULATION_CONFIG)
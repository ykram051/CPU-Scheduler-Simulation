"""
Configuration file for CPU Scheduler Simulation
This file allows for easy customization of simulation parameters
"""

# Simulation Parameters
SIMULATION_CONFIG = {
    # Process Generation
    "random_process": {
        "min_arrival_time": 0,
        "max_arrival_time": 50,
        "min_burst_time": 1,
        "max_burst_time": 20,
        "min_priority": 1,
        "max_priority": 10,
        "default_count": 10
    },
    
    # Algorithm Parameters
    "round_robin": {
        "default_quantum": 2,
        "quantum_options": [1, 2, 4, 8]
    },
    "priority_round_robin": {
        "default_quantum": 2,
        "quantum_options": [1, 2, 4]
    },
    "multilevel_feedback_queue": {
        "num_queues": 3,
        "base_quantum": 2,
        "aging_threshold": 10  # Time units after which a process priority may be increased
    },
    
    # Visualization Settings
    "visualization": {
        "gantt_chart_height": 0.6,
        "timeline_height": 0.3,
        "color_palette": "viridis",  # Options: viridis, plasma, inferno, magma, cividis
        "show_pid_labels": True,
        "show_metrics_in_chart": True,
        "export_formats": ["png", "pdf"],
        "dpi": 300
    },
    
    # Advanced Metrics
    "metrics": {
        "calculate_fairness": True,
        "calculate_starvation_metrics": True,
        "normalize_by_burst_time": False
    },
    
    # UI Settings
    "ui": {
        "theme": "light",  # Options: light, dark
        "show_advanced_options": False,
        "max_processes_in_ui": 20,
        "auto_refresh_charts": True,
        "show_execution_trace": True
    }
}

def get_config(section=None, key=None):
    """
    Get configuration values
    
    Args:
        section: Configuration section name
        key: Specific configuration key
        
    Returns:
        Configuration value or section dictionary
    """
    if section is None:
        return SIMULATION_CONFIG
    
    if key is None:
        return SIMULATION_CONFIG.get(section, {})
    
    return SIMULATION_CONFIG.get(section, {}).get(key, None)

def update_config(section, key, value):
    """
    Update a configuration value
    
    Args:
        section: Configuration section name
        key: Configuration key to update
        value: New value
    """
    if section in SIMULATION_CONFIG and key in SIMULATION_CONFIG[section]:
        SIMULATION_CONFIG[section][key] = value
        return True
    return False
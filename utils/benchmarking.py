"""
Benchmarking utilities for the distributed supervised learning platform.
Provides tools for automated performance testing and resource optimization.
"""
import ray
import time
import json
import os
import psutil
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from concurrent.futures import ThreadPoolExecutor
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """Provides utilities for benchmarking and optimizing distributed training."""
    
    def __init__(self, metrics_dir: str = "benchmark_results"):
        """
        Initialize the benchmarking utility.
        
        Args:
            metrics_dir: Directory to store benchmark results
        """
        self.metrics_dir = metrics_dir
        os.makedirs(metrics_dir, exist_ok=True)
        self.benchmark_history = []
        
    def benchmark_training(self, training_function: Callable, 
                           dataset_sizes: List[int], 
                           model_configs: List[Dict[str, Any]],
                           repeats: int = 3,
                           **training_kwargs) -> Dict[str, Any]:
        """
        Benchmark training performance with various dataset sizes and model configurations.
        
        Args:
            training_function: Function that performs the training
            dataset_sizes: List of dataset sizes to test
            model_configs: List of model configurations to test
            repeats: Number of times to repeat each benchmark for statistical significance
            **training_kwargs: Additional arguments to pass to the training function
            
        Returns:
            Dictionary of benchmark results
        """
        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "dataset_sizes": dataset_sizes,
            "model_configs": model_configs,
            "repeats": repeats,
            "results": []
        }
        
        # Ensure Ray is initialized
        if not ray.is_initialized():
            ray.init()
            
        # Start benchmarking
        logger.info(f"Starting benchmark with {len(dataset_sizes)} dataset sizes x {len(model_configs)} models x {repeats} repeats")
        
        # For each dataset size
        for size in dataset_sizes:
            size_results = {
                "dataset_size": size,
                "models": []
            }
            
            # For each model configuration
            for model_config in model_configs:
                model_name = model_config.get('name', str(model_config))
                model_results = {
                    "model_name": model_name,
                    "runs": []
                }
                
                # Repeat multiple times for statistical significance
                for i in range(repeats):
                    logger.info(f"Running benchmark: Size={size}, Model={model_name}, Run {i+1}/{repeats}")
                    
                    # Capture system metrics before
                    system_before = self._capture_system_status()
                    
                    # Execute the training function and measure time
                    start_time = time.time()
                    try:
                        # Add dataset size to kwargs
                        kwargs = training_kwargs.copy()
                        kwargs['dataset_size'] = size
                        # Run the training function
                        result = training_function([model_config], **kwargs)
                        success = True
                    except Exception as e:
                        logger.error(f"Error during benchmark: {str(e)}")
                        result = {"error": str(e)}
                        success = False
                    
                    # Calculate execution time
                    execution_time = time.time() - start_time
                    
                    # Capture system metrics after
                    system_after = self._capture_system_status()
                    
                    # Record results
                    run_result = {
                        "run_id": i,
                        "success": success,
                        "execution_time": execution_time,
                        "system_before": system_before,
                        "system_after": system_after,
                        "result": result
                    }
                    
                    model_results["runs"].append(run_result)
                
                # Calculate statistics
                execution_times = [run["execution_time"] for run in model_results["runs"] if run["success"]]
                if execution_times:
                    model_results["stats"] = {
                        "mean_time": np.mean(execution_times),
                        "std_time": np.std(execution_times),
                        "min_time": np.min(execution_times),
                        "max_time": np.max(execution_times),
                        "success_rate": sum(run["success"] for run in model_results["runs"]) / len(model_results["runs"])
                    }
                
                size_results["models"].append(model_results)
            
            results["results"].append(size_results)
        
        # Save benchmark results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{timestamp}.json"
        filepath = os.path.join(self.metrics_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {filepath}")
        self.benchmark_history.append(results)
        
        return results
    
    def _capture_system_status(self) -> Dict[str, Any]:
        """Capture current system status for benchmark comparison."""
        try:
            # CPU info
            cpu_percent = psutil.cpu_percent(interval=0.5)
            
            # Memory info
            memory = psutil.virtual_memory()
            
            # Ray info if available
            ray_info = {}
            if ray.is_initialized():
                ray_info = {
                    "available_resources": ray.available_resources(),
                    "cluster_resources": ray.cluster_resources()
                }
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024 * 1024 * 1024),
                "ray_info": ray_info,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error capturing system status: {str(e)}")
            return {"error": str(e)}
    
    def get_optimal_configuration(self) -> Dict[str, Any]:
        """
        Analyze benchmark history to determine the optimal configuration.
        
        Returns:
            Dictionary with optimal configuration recommendations
        """
        if not self.benchmark_history:
            return {"error": "No benchmark history available"}
        
        # Prepare data for analysis
        all_configs = []
        
        for benchmark in self.benchmark_history:
            for size_result in benchmark["results"]:
                dataset_size = size_result["dataset_size"]
                
                for model_result in size_result["models"]:
                    if "stats" in model_result:
                        all_configs.append({
                            "dataset_size": dataset_size,
                            "model_name": model_result["model_name"],
                            "mean_time": model_result["stats"]["mean_time"],
                            "success_rate": model_result["stats"]["success_rate"]
                        })
        
        if not all_configs:
            return {"error": "No valid configurations found in benchmark history"}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(all_configs)
        
        # Find optimal configurations
        fastest_config = df.loc[df["mean_time"].idxmin()]
        most_reliable_config = df.loc[df["success_rate"].idxmax()]
        
        # Get optimal config by dataset size
        optimal_by_size = {}
        for size in df["dataset_size"].unique():
            size_df = df[df["dataset_size"] == size]
            
            # Simple scoring: normalize time (lower is better) and success rate (higher is better)
            # and combine them with a weight
            size_df["time_score"] = 1 - (size_df["mean_time"] - size_df["mean_time"].min()) / (size_df["mean_time"].max() - size_df["mean_time"].min() + 1e-10)
            size_df["combined_score"] = 0.7 * size_df["time_score"] + 0.3 * size_df["success_rate"]
            
            best_config = size_df.loc[size_df["combined_score"].idxmax()]
            optimal_by_size[str(size)] = {
                "model_name": best_config["model_name"],
                "mean_time": best_config["mean_time"],
                "success_rate": best_config["success_rate"],
                "score": best_config["combined_score"]
            }
        
        return {
            "fastest_overall": {
                "model_name": fastest_config["model_name"],
                "dataset_size": fastest_config["dataset_size"],
                "mean_time": fastest_config["mean_time"]
            },
            "most_reliable": {
                "model_name": most_reliable_config["model_name"],
                "dataset_size": most_reliable_config["dataset_size"],
                "success_rate": most_reliable_config["success_rate"]
            },
            "optimal_by_size": optimal_by_size
        }


class RayTaskTracker:
    """Provides tools for tracking Ray tasks and their resource consumption."""
    
    def __init__(self, tracking_interval: float = 1.0, max_tasks: int = 500):
        """
        Initialize the Ray task tracker.
        
        Args:
            tracking_interval: Interval in seconds between tracking updates
            max_tasks: Maximum number of tasks to keep in history
        """
        self.tracking_interval = tracking_interval
        self.max_tasks = max_tasks
        self.tracking = False
        self.task_history = []
        self._executor = None
    
    def start_tracking(self):
        """Start tracking Ray tasks."""
        if not ray.is_initialized():
            ray.init()
        
        if self.tracking:
            logger.warning("Task tracking is already running")
            return
        
        self.tracking = True
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._executor.submit(self._tracking_loop)
        logger.info("Ray task tracking started")
    
    def stop_tracking(self):
        """Stop tracking Ray tasks."""
        self.tracking = False
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None
        logger.info("Ray task tracking stopped")
    
    def _tracking_loop(self):
        """Main tracking loop."""
        while self.tracking:
            try:
                # Get Ray task info
                # This is an internal API and may change in future Ray versions
                tasks = ray._private.state.global_state.task_table()
                
                # Process tasks
                for task_id, task_info in tasks.items():
                    # Convert binary protobuf to Python dict
                    if hasattr(task_info, "task_spec"):
                        # Skip tasks that are already in history
                        if any(t["task_id"] == task_id for t in self.task_history):
                            continue
                        
                        # Extract task info
                        task_data = {
                            "task_id": task_id,
                            "timestamp": time.time(),
                            "state": task_info.state_name,
                            "function_name": task_info.function_name if hasattr(task_info, "function_name") else "unknown",
                            "resources": {}
                        }
                        
                        # Extract resource requirements if available
                        if hasattr(task_info, "required_resources"):
                            for resource, amount in task_info.required_resources.items():
                                task_data["resources"][resource] = amount
                        
                        # Add to history
                        self.task_history.append(task_data)
                        
                        # Trim history if needed
                        if len(self.task_history) > self.max_tasks:
                            self.task_history = self.task_history[-self.max_tasks:]
                
            except Exception as e:
                logger.error(f"Error in task tracking: {str(e)}")
            
            # Sleep for interval
            time.sleep(self.tracking_interval)
    
    def get_task_stats(self) -> Dict[str, Any]:
        """
        Get statistics about tracked tasks.
        
        Returns:
            Dictionary with task statistics
        """
        if not self.task_history:
            return {"error": "No task history available"}
        
        # Count tasks by state
        states = {}
        for task in self.task_history:
            state = task.get("state", "unknown")
            states[state] = states.get(state, 0) + 1
        
        # Count tasks by function name
        functions = {}
        for task in self.task_history:
            func = task.get("function_name", "unknown")
            functions[func] = functions.get(func, 0) + 1
        
        # Calculate average resource usage
        resources = {}
        for task in self.task_history:
            for resource, amount in task.get("resources", {}).items():
                if resource not in resources:
                    resources[resource] = []
                resources[resource].append(amount)
        
        avg_resources = {}
        for resource, amounts in resources.items():
            avg_resources[resource] = sum(amounts) / len(amounts)
        
        return {
            "total_tasks": len(self.task_history),
            "states": states,
            "functions": functions,
            "avg_resources": avg_resources,
            "is_tracking": self.tracking
        }


class ResourceForecaster:
    """Provides tools for forecasting resource usage based on historical data."""
    
    def __init__(self, history_window: int = 50):
        """
        Initialize the resource forecaster.
        
        Args:
            history_window: Number of historical data points to keep
        """
        self.history_window = history_window
        self.cpu_history = []
        self.memory_history = []
        self.timestamp_history = []
    
    def add_datapoint(self, cpu_percent: float, memory_percent: float):
        """
        Add a new datapoint to the history.
        
        Args:
            cpu_percent: Current CPU usage percentage
            memory_percent: Current memory usage percentage
        """
        self.cpu_history.append(cpu_percent)
        self.memory_history.append(memory_percent)
        self.timestamp_history.append(time.time())
        
        # Trim history if needed
        if len(self.cpu_history) > self.history_window:
            self.cpu_history = self.cpu_history[-self.history_window:]
            self.memory_history = self.memory_history[-self.history_window:]
            self.timestamp_history = self.timestamp_history[-self.history_window:]
    
    def forecast(self, minutes_ahead: int = 5) -> Dict[str, Any]:
        """
        Forecast resource usage based on historical data.
        
        Args:
            minutes_ahead: Number of minutes to forecast ahead
            
        Returns:
            Dictionary with forecast data
        """
        if len(self.cpu_history) < 10:
            return {
                "error": "Not enough history for forecasting",
                "required_points": 10,
                "current_points": len(self.cpu_history)
            }
        
        try:
            # Simple linear regression for forecasting
            x = np.array(self.timestamp_history)
            cpu_y = np.array(self.cpu_history)
            mem_y = np.array(self.memory_history)
            
            # Normalize x to be relative to the first timestamp
            x = x - x[0]
            
            # Fit linear models
            cpu_coeffs = np.polyfit(x, cpu_y, 1)
            mem_coeffs = np.polyfit(x, mem_y, 1)
            
            cpu_model = np.poly1d(cpu_coeffs)
            mem_model = np.poly1d(mem_coeffs)
            
            # Forecast
            seconds_ahead = minutes_ahead * 60
            future_time = x[-1] + seconds_ahead
            
            cpu_forecast = float(cpu_model(future_time))
            mem_forecast = float(mem_model(future_time))
            
            # Cap forecasts to reasonable ranges
            cpu_forecast = max(0, min(100, cpu_forecast))
            mem_forecast = max(0, min(100, mem_forecast))
            
            # Current trends (percentage points per minute)
            cpu_trend = 60 * cpu_coeffs[0]  # Convert from per-second to per-minute
            mem_trend = 60 * mem_coeffs[0]
            
            return {
                "cpu_forecast": cpu_forecast,
                "memory_forecast": mem_forecast,
                "cpu_trend": cpu_trend,
                "memory_trend": mem_trend,
                "minutes_ahead": minutes_ahead,
                "forecast_timestamp": time.time() + seconds_ahead
            }
        except Exception as e:
            logger.error(f"Error in forecasting: {str(e)}")
            return {"error": str(e)}


class ResourceAlertSystem:
    """Provides tools for alerting when resource usage exceeds thresholds."""
    
    def __init__(self):
        """Initialize the resource alert system."""
        self.alerts = []
        self.thresholds = {
            "cpu_percent": 80.0,  # Alert when CPU usage exceeds 80%
            "memory_percent": 80.0,  # Alert when memory usage exceeds 80%
            "disk_percent": 90.0,  # Alert when disk usage exceeds 90%
            "ray_cpu_percent": 90.0,  # Alert when Ray CPU usage exceeds 90%
            "ray_memory_percent": 90.0  # Alert when Ray memory usage exceeds 90%
        }
    
    def set_threshold(self, resource: str, threshold: float):
        """
        Set a threshold for a resource.
        
        Args:
            resource: Resource name
            threshold: Threshold value (percentage)
        """
        if resource in self.thresholds:
            self.thresholds[resource] = threshold
        else:
            logger.warning(f"Unknown resource: {resource}")
    
    def check_resources(self, system_resources: Dict[str, Any], ray_status: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check resources against thresholds and generate alerts.
        
        Args:
            system_resources: System resources data
            ray_status: Ray cluster status data
            
        Returns:
            List of alerts generated in this check
        """
        new_alerts = []
        
        # Check system resources
        try:
            # CPU check
            if system_resources.get("cpu", {}).get("percent", 0) > self.thresholds["cpu_percent"]:
                new_alerts.append({
                    "timestamp": time.time(),
                    "resource": "cpu",
                    "value": system_resources["cpu"]["percent"],
                    "threshold": self.thresholds["cpu_percent"],
                    "message": f"CPU usage ({system_resources['cpu']['percent']}%) exceeds threshold ({self.thresholds['cpu_percent']}%)"
                })
            
            # Memory check
            if system_resources.get("memory", {}).get("percent", 0) > self.thresholds["memory_percent"]:
                new_alerts.append({
                    "timestamp": time.time(),
                    "resource": "memory",
                    "value": system_resources["memory"]["percent"],
                    "threshold": self.thresholds["memory_percent"],
                    "message": f"Memory usage ({system_resources['memory']['percent']}%) exceeds threshold ({self.thresholds['memory_percent']}%)"
                })
            
            # Disk check
            if system_resources.get("disk", {}).get("percent", 0) > self.thresholds["disk_percent"]:
                new_alerts.append({
                    "timestamp": time.time(),
                    "resource": "disk",
                    "value": system_resources["disk"]["percent"],
                    "threshold": self.thresholds["disk_percent"],
                    "message": f"Disk usage ({system_resources['disk']['percent']}%) exceeds threshold ({self.thresholds['disk_percent']}%)"
                })
        except Exception as e:
            logger.error(f"Error checking system resources: {str(e)}")
        
        # Check Ray resources
        try:
            if ray_status.get("initialized", False):
                # Calculate Ray CPU usage percentage
                total_cpu = ray_status.get("total_resources", {}).get("CPU", 0)
                used_cpu = ray_status.get("used_resources", {}).get("CPU", 0)
                
                if total_cpu > 0:
                    ray_cpu_percent = (used_cpu / total_cpu) * 100
                    
                    if ray_cpu_percent > self.thresholds["ray_cpu_percent"]:
                        new_alerts.append({
                            "timestamp": time.time(),
                            "resource": "ray_cpu",
                            "value": ray_cpu_percent,
                            "threshold": self.thresholds["ray_cpu_percent"],
                            "message": f"Ray CPU usage ({ray_cpu_percent:.1f}%) exceeds threshold ({self.thresholds['ray_cpu_percent']}%)"
                        })
                
                # Calculate Ray memory usage percentage
                total_memory = ray_status.get("total_resources", {}).get("memory", 0)
                used_memory = ray_status.get("used_resources", {}).get("memory", 0)
                
                if total_memory > 0:
                    ray_memory_percent = (used_memory / total_memory) * 100
                    
                    if ray_memory_percent > self.thresholds["ray_memory_percent"]:
                        new_alerts.append({
                            "timestamp": time.time(),
                            "resource": "ray_memory",
                            "value": ray_memory_percent,
                            "threshold": self.thresholds["ray_memory_percent"],
                            "message": f"Ray memory usage ({ray_memory_percent:.1f}%) exceeds threshold ({self.thresholds['ray_memory_percent']}%)"
                        })
        except Exception as e:
            logger.error(f"Error checking Ray resources: {str(e)}")
        
        # Add new alerts to the history
        self.alerts.extend(new_alerts)
        
        return new_alerts
    
    def get_alerts(self, max_count: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent alerts.
        
        Args:
            max_count: Maximum number of alerts to return
            
        Returns:
            List of recent alerts
        """
        # Return most recent alerts first
        return sorted(self.alerts, key=lambda x: x["timestamp"], reverse=True)[:max_count]
    
    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts = []

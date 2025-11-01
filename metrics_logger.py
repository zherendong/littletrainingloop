"""
Metrics logger for saving training and evaluation metrics to JSON files.

This provides a Neptune-like interface but saves to local JSON files instead.
Each experiment gets its own directory with structured metrics.
"""

import json
import os
from pathlib import Path
from typing import Any
from collections import defaultdict


class MetricsLogger:
    """
    Logger that saves metrics to JSON files.
    
    Directory structure:
        experiments/
            {experiment_name}/
                config.json          # Experiment configuration
                metrics.jsonl        # All metrics (one JSON object per line)
                summary.json         # Final summary statistics
    """
    
    def __init__(self, experiment_name: str, base_dir: str = "experiments/metrics"):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.experiment_dir = self.base_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.experiment_dir / "metrics.jsonl"
        self.config_file = self.experiment_dir / "config.json"
        self.summary_file = self.experiment_dir / "summary.json"
        
        # In-memory storage for aggregation
        self.all_metrics = defaultdict(list)
        
        print(f"📊 Metrics will be saved to: {self.experiment_dir}")
    
    def log_config(self, config: dict[str, Any]):
        """Save experiment configuration."""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        print(f"✓ Config saved to {self.config_file}")
    
    def log_metric(self, name: str, value: float, step: int | None = None):
        """
        Log a single metric value.
        
        Args:
            name: Metric name (e.g., "train/loss", "eval/loss")
            value: Metric value
            step: Training step number
        """
        metric_entry = {
            "name": name,
            "value": float(value),
            "step": step,
        }
        
        # Append to JSONL file (one JSON object per line)
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metric_entry) + '\n')
        
        # Store in memory for summary
        self.all_metrics[name].append({
            "step": step,
            "value": float(value),
        })
    
    def log_metrics(self, metrics: dict[str, float], step: int | None = None, prefix: str = ""):
        """
        Log multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Training step number
            prefix: Prefix to add to metric names (e.g., "train/", "eval/")
        """
        for name, value in metrics.items():
            full_name = f"{prefix}{name}" if prefix else name
            self.log_metric(full_name, value, step)
    
    def save_summary(self):
        """Save summary statistics (final values, min, max, etc.)."""
        summary = {}
        
        for name, values in self.all_metrics.items():
            if not values:
                continue
            
            value_list = [v["value"] for v in values]
            summary[name] = {
                "final": values[-1]["value"],
                "min": min(value_list),
                "max": max(value_list),
                "count": len(values),
                "final_step": values[-1]["step"],
            }
        
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Summary saved to {self.summary_file}")
    
    def get_metrics_by_name(self, name: str) -> list[dict]:
        """Get all logged values for a specific metric."""
        return self.all_metrics.get(name, [])
    
    def close(self):
        """Finalize logging and save summary."""
        self.save_summary()
        print(f"✓ Metrics logging completed for {self.experiment_name}")


class NeptuneCompatibleLogger:
    """
    Neptune-compatible interface that wraps MetricsLogger.
    
    This allows drop-in replacement of Neptune with local logging.
    Usage:
        logger = NeptuneCompatibleLogger("experiment_name")
        logger["train/loss"].append(0.5, step=0)
        logger["config"] = {"lr": 0.001}
    """
    
    def __init__(self, experiment_name: str, base_dir: str = "experiments/metrics"):
        self.logger = MetricsLogger(experiment_name, base_dir)
        self._config = {}
    
    def __getitem__(self, key: str):
        """Return a metric series object."""
        return MetricSeries(self.logger, key)
    
    def __setitem__(self, key: str, value: Any):
        """Set a config value or metadata."""
        if key == "config":
            self._config = value
            self.logger.log_config(value)
        else:
            # Treat as a single-value metric
            self._config[key] = value
            if self._config:
                self.logger.log_config(self._config)
    
    def stop(self):
        """Finalize logging."""
        self.logger.close()


class MetricSeries:
    """Represents a single metric series (e.g., train/loss)."""
    
    def __init__(self, logger: MetricsLogger, name: str):
        self.logger = logger
        self.name = name
    
    def append(self, value: float, step: int | None = None):
        """Append a value to this metric series."""
        self.logger.log_metric(self.name, value, step)


def load_metrics(experiment_name: str, base_dir: str = "experiments/metrics") -> dict:
    """
    Load all metrics from an experiment.
    
    Returns:
        Dictionary with:
            - "config": experiment configuration
            - "metrics": dict of metric_name -> list of {step, value}
            - "summary": summary statistics
    """
    experiment_dir = Path(base_dir) / experiment_name
    
    # Load config
    config_file = experiment_dir / "config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Load metrics from JSONL
    metrics = defaultdict(list)
    metrics_file = experiment_dir / "metrics.jsonl"
    with open(metrics_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            metrics[entry["name"]].append({
                "step": entry["step"],
                "value": entry["value"],
            })
    
    # Load summary
    summary_file = experiment_dir / "summary.json"
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    return {
        "config": config,
        "metrics": dict(metrics),
        "summary": summary,
    }


def list_experiments(base_dir: str = "experiments/metrics") -> list[str]:
    """List all available experiments."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    experiments = []
    for exp_dir in base_path.iterdir():
        if exp_dir.is_dir() and (exp_dir / "metrics.jsonl").exists():
            experiments.append(exp_dir.name)
    
    return sorted(experiments)


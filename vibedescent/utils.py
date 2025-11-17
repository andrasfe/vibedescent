"""
Utility functions for Vibe Descent framework.
"""

import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import asdict

from .core import Candidate, EvalResult


class Logger:
    """Simple logger for training progress."""
    
    def __init__(self, log_dir: Optional[str] = None, verbose: bool = True):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save logs (if None, no file logging)
            verbose: Whether to print to console
        """
        self.log_dir = Path(log_dir) if log_dir else None
        self.verbose = verbose
        self.log_data = []
        
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = self.log_dir / f"vibe_descent_{int(time.time())}.jsonl"
    
    def log_iteration(
        self,
        iteration: int,
        candidate: Candidate,
        state_dict: Dict[str, Any]
    ):
        """Log an iteration."""
        entry = {
            'iteration': iteration,
            'timestamp': time.time(),
            'operator': {
                'type': candidate.operator.type,
                'params': candidate.operator.params,
            },
            'evaluation': {
                'feasible': candidate.evaluation.feasible,
                'objective': candidate.evaluation.objective_value,
                'gap_pct': candidate.evaluation.gap_pct,
                'runtime_ms': candidate.evaluation.runtime_ms,
            },
            'loss': candidate.loss,
            'state': state_dict,
        }
        
        self.log_data.append(entry)
        
        if self.log_dir:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
    
    def save_summary(self, filename: str = 'summary.json'):
        """Save summary of entire run."""
        if not self.log_dir:
            return
        
        summary = {
            'total_iterations': len(self.log_data),
            'start_time': self.log_data[0]['timestamp'] if self.log_data else None,
            'end_time': self.log_data[-1]['timestamp'] if self.log_data else None,
            'best_objective': max(
                entry['evaluation']['objective'] for entry in self.log_data
            ) if self.log_data else None,
            'best_gap': min(
                entry['evaluation']['gap_pct'] for entry in self.log_data
            ) if self.log_data else None,
            'operators_used': self._count_operators(),
        }
        
        with open(self.log_dir / filename, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _count_operators(self) -> Dict[str, int]:
        """Count operator usage."""
        counts = {}
        for entry in self.log_data:
            op_type = entry['operator']['type']
            counts[op_type] = counts.get(op_type, 0) + 1
        return counts


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def format_number(n: float, decimals: int = 2) -> str:
    """Format large numbers with K/M/B suffixes."""
    if abs(n) < 1000:
        return f"{n:.{decimals}f}"
    elif abs(n) < 1_000_000:
        return f"{n/1000:.{decimals}f}K"
    elif abs(n) < 1_000_000_000:
        return f"{n/1_000_000:.{decimals}f}M"
    else:
        return f"{n/1_000_000_000:.{decimals}f}B"


def compute_progress_metrics(history: List[Candidate]) -> Dict[str, Any]:
    """
    Compute progress metrics from history.
    
    Returns metrics like improvement rate, convergence, etc.
    """
    if not history:
        return {}
    
    losses = [c.loss for c in history]
    objectives = [c.evaluation.objective_value for c in history]
    gaps = [c.evaluation.gap_pct for c in history if c.evaluation.gap_pct < float('inf')]
    
    metrics = {
        'iterations': len(history),
        'best_loss': min(losses),
        'final_loss': losses[-1],
        'loss_improvement': losses[0] - losses[-1],
        'loss_improvement_pct': (losses[0] - losses[-1]) / losses[0] * 100 if losses[0] > 0 else 0,
        'best_objective': max(objectives),
        'final_objective': objectives[-1],
    }
    
    if gaps:
        metrics['best_gap'] = min(gaps)
        metrics['final_gap'] = gaps[-1]
    
    # Convergence rate (over last 10 iterations)
    if len(losses) >= 10:
        recent_losses = losses[-10:]
        convergence_rate = (recent_losses[0] - recent_losses[-1]) / 10
        metrics['convergence_rate'] = convergence_rate
        
        # Stability (variance of recent losses)
        mean_loss = sum(recent_losses) / len(recent_losses)
        variance = sum((l - mean_loss) ** 2 for l in recent_losses) / len(recent_losses)
        metrics['stability'] = 1.0 / (1.0 + variance)
    
    return metrics


def save_solution_to_file(
    solution: Any,
    evaluation: EvalResult,
    filepath: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save solution to file.
    
    Args:
        solution: Solution object
        evaluation: Evaluation result
        filepath: Path to save to
        metadata: Additional metadata to include
    """
    data = {
        'solution': solution.to_dict() if hasattr(solution, 'to_dict') else str(solution),
        'evaluation': {
            'feasible': evaluation.feasible,
            'objective_value': evaluation.objective_value,
            'gap_pct': evaluation.gap_pct,
            'upper_bound': evaluation.upper_bound,
            'lower_bound': evaluation.lower_bound,
            'runtime_ms': evaluation.runtime_ms,
            'metadata': evaluation.metadata,
        },
        'metadata': metadata or {},
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_solution_from_file(filepath: str) -> Dict[str, Any]:
    """Load solution from file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def print_comparison_table(
    solutions: List[tuple],
    headers: List[str] = None
):
    """
    Print a comparison table of solutions.
    
    Args:
        solutions: List of (name, objective, gap, time) tuples
        headers: Column headers
    """
    if not solutions:
        return
    
    headers = headers or ['Method', 'Objective', 'Gap (%)', 'Time (s)']
    
    # Compute column widths
    widths = [max(len(h), 15) for h in headers]
    for sol in solutions:
        widths[0] = max(widths[0], len(str(sol[0])))
    
    # Print header
    print()
    print(' | '.join(h.ljust(w) for h, w in zip(headers, widths)))
    print('-|-'.join('-' * w for w in widths))
    
    # Print rows
    for sol in solutions:
        name = str(sol[0]).ljust(widths[0])
        obj = f"{sol[1]:.2f}".rjust(widths[1])
        gap = f"{sol[2]:.2f}".rjust(widths[2])
        time_str = f"{sol[3]:.2f}".rjust(widths[3])
        print(f"{name} | {obj} | {gap} | {time_str}")
    print()


def create_progress_plot(
    history: List[Candidate],
    save_path: Optional[str] = None,
    metrics: List[str] = ['loss', 'objective', 'gap']
):
    """
    Create a progress plot (requires matplotlib).
    
    Args:
        history: Training history
        save_path: Path to save plot (if None, shows interactive)
        metrics: Metrics to plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    
    iterations = list(range(len(history)))
    
    for ax, metric in zip(axes, metrics):
        if metric == 'loss':
            values = [c.loss for c in history]
            ax.set_ylabel('Loss')
        elif metric == 'objective':
            values = [c.evaluation.objective_value for c in history]
            ax.set_ylabel('Objective Value')
        elif metric == 'gap':
            values = [c.evaluation.gap_pct for c in history if c.evaluation.gap_pct < float('inf')]
            iterations_gap = [i for i, c in enumerate(history) if c.evaluation.gap_pct < float('inf')]
            ax.plot(iterations_gap, values, marker='o', linewidth=2)
            ax.set_ylabel('Gap (%)')
            ax.set_xlabel('Iteration')
            continue
        
        ax.plot(iterations, values, marker='o', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


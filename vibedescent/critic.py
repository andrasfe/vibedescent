"""
Critic implementations for evaluating and selecting candidates.
"""

from typing import List, Dict, Any
from .core import Critic as BaseCritic, Candidate


class SimpleCritic(BaseCritic):
    """
    Simple critic that selects by lowest loss.
    
    Provides basic progress analysis.
    """
    
    def select_best(
        self,
        candidates: List[Candidate],
        context: Dict[str, Any]
    ) -> Candidate:
        """Select candidate with lowest loss."""
        if not candidates:
            raise ValueError("Cannot select from empty candidate list")
        
        return min(candidates, key=lambda c: c.loss)
    
    def analyze_progress(
        self,
        history: List[Candidate],
        current_best: Candidate
    ) -> Dict[str, Any]:
        """
        Analyze optimization progress.
        
        Returns metrics about improvement rate, stability, etc.
        """
        if not history:
            return {
                'improvement_rate': 0.0,
                'stability': 1.0,
                'stagnation_count': 0
            }
        
        # Compute improvement rate over last N iterations
        window = min(10, len(history))
        recent = history[-window:]
        
        if len(recent) < 2:
            improvement_rate = 0.0
        else:
            loss_deltas = [
                recent[i-1].loss - recent[i].loss
                for i in range(1, len(recent))
            ]
            improvement_rate = sum(loss_deltas) / len(loss_deltas)
        
        # Compute stability (low variance is stable)
        if len(recent) >= 3:
            losses = [c.loss for c in recent]
            mean_loss = sum(losses) / len(losses)
            variance = sum((l - mean_loss) ** 2 for l in losses) / len(losses)
            stability = 1.0 / (1.0 + variance)
        else:
            stability = 1.0
        
        # Count stagnation (no improvement)
        stagnation_count = 0
        for i in range(len(history) - 1, 0, -1):
            if history[i].loss >= history[i-1].loss:
                stagnation_count += 1
            else:
                break
        
        return {
            'improvement_rate': improvement_rate,
            'stability': stability,
            'stagnation_count': stagnation_count,
            'best_loss': current_best.loss,
            'recent_mean_loss': sum(c.loss for c in recent) / len(recent)
        }


class DiversityCritic(SimpleCritic):
    """
    Critic that balances loss minimization with diversity.
    
    Occasionally selects diverse candidates to avoid local minima.
    """
    
    def __init__(self, diversity_rate: float = 0.1):
        """
        Initialize diversity critic.
        
        Args:
            diversity_rate: Probability of selecting diverse candidate instead of best
        """
        self.diversity_rate = diversity_rate
        self.selection_count = 0
    
    def select_best(
        self,
        candidates: List[Candidate],
        context: Dict[str, Any]
    ) -> Candidate:
        """Select best or diverse candidate."""
        import random
        
        self.selection_count += 1
        
        # Occasionally pick a diverse candidate
        if random.random() < self.diversity_rate and len(candidates) > 1:
            # Sort by loss
            sorted_candidates = sorted(candidates, key=lambda c: c.loss)
            
            # Pick from top 50% but not the absolute best
            diverse_pool = sorted_candidates[1:len(sorted_candidates)//2 + 1]
            if diverse_pool:
                return random.choice(diverse_pool)
        
        # Default: select best
        return super().select_best(candidates, context)


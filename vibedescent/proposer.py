"""
Proposer implementations for generating candidate operators.
"""

from typing import List, Dict, Any, Callable, Optional, Generic, TypeVar
import random

from .core import Proposer as BaseProposer, Operator, Candidate, Problem, Solution

SolutionType = TypeVar('SolutionType', bound=Solution)


class AdaptiveProposer(BaseProposer, Generic[SolutionType]):
    """
    Adaptive proposer that samples from a pool of operators.
    
    Adjusts operator parameters based on trust region and can use
    momentum to prefer operator classes that have been successful.
    """
    
    def __init__(
        self,
        operator_pool: List[Dict[str, Any]],
        use_momentum: bool = True,
        exploration_rate: float = 0.2
    ):
        """
        Initialize adaptive proposer.
        
        Args:
            operator_pool: List of base operator specifications
            use_momentum: Whether to bias toward successful operator types
            exploration_rate: Fraction of pure exploration vs exploitation
        """
        self.operator_pool = operator_pool
        self.use_momentum = use_momentum
        self.exploration_rate = exploration_rate
        self.operator_scores: Dict[str, float] = {}
        self.rng = random.Random()
    
    def propose_operators(
        self,
        problem: Problem,
        current_solution: SolutionType,
        history: List[Candidate],
        trust_region: float,
        k: int = 6
    ) -> List[Operator]:
        """Generate k operators with trust-region adjusted parameters."""
        
        # Update momentum from history
        if self.use_momentum and history:
            self._update_momentum(history[-10:])  # Last 10 candidates
        
        operators = []
        for _ in range(k):
            # Exploration vs exploitation
            if self.rng.random() < self.exploration_rate:
                base_op = self.rng.choice(self.operator_pool)
            else:
                base_op = self._sample_by_momentum()
            
            # Create operator with jittered parameters
            op = self._jitter_operator(base_op, trust_region)
            operators.append(op)
        
        return operators
    
    def _update_momentum(self, recent_history: List[Candidate]):
        """Update operator type scores based on recent performance."""
        for candidate in recent_history:
            op_type = candidate.operator.type
            # Lower loss is better, so we invert it for scoring
            score = 1.0 / (1.0 + candidate.loss)
            
            if op_type not in self.operator_scores:
                self.operator_scores[op_type] = score
            else:
                # Exponential moving average
                self.operator_scores[op_type] = (
                    0.7 * self.operator_scores[op_type] + 0.3 * score
                )
    
    def _sample_by_momentum(self) -> Dict[str, Any]:
        """Sample operator biased by momentum scores."""
        if not self.operator_scores:
            return self.rng.choice(self.operator_pool)
        
        # Weighted sampling
        types_in_pool = [op['type'] for op in self.operator_pool]
        weights = [self.operator_scores.get(op['type'], 1.0) for op in self.operator_pool]
        total = sum(weights)
        
        if total == 0:
            return self.rng.choice(self.operator_pool)
        
        r = self.rng.random() * total
        cumsum = 0
        for op, w in zip(self.operator_pool, weights):
            cumsum += w
            if r <= cumsum:
                return op
        
        return self.operator_pool[-1]
    
    def _jitter_operator(self, base_op: Dict[str, Any], trust_region: float) -> Operator:
        """Apply trust-region based jittering to operator parameters."""
        op_dict = base_op.copy()
        op_type = op_dict.pop('type')
        params = op_dict.copy()
        
        # Jitter numeric parameters
        for key, value in params.items():
            if isinstance(value, (int, float)) and key != 'type':
                # Apply multiplicative jitter scaled by trust region
                jitter_factor = 0.5 + trust_region * self.rng.random()
                params[key] = type(value)(value * jitter_factor)
        
        return Operator(type=op_type, params=params)


class ModelProposer(BaseProposer, Generic[SolutionType]):
    """
    Proposer that uses a reasoning model (LLM) to generate operators.
    
    Falls back to adaptive proposer if model fails.
    """
    
    def __init__(
        self,
        model_fn: Callable[[List[Candidate], Dict[str, Any]], List[Dict[str, Any]]],
        fallback_proposer: Optional[BaseProposer] = None,
        operator_pool: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize model-based proposer.
        
        Args:
            model_fn: Function that takes (history, context) and returns operator dicts
            fallback_proposer: Proposer to use if model fails
            operator_pool: Operator pool for fallback
        """
        self.model_fn = model_fn
        self.fallback_proposer = fallback_proposer or (
            AdaptiveProposer(operator_pool) if operator_pool else None
        )
    
    def propose_operators(
        self,
        problem: Problem,
        current_solution: SolutionType,
        history: List[Candidate],
        trust_region: float,
        k: int = 6
    ) -> List[Operator]:
        """Generate operators using model, with fallback."""
        
        try:
            context = problem.get_context()
            context['trust_region'] = trust_region
            
            operator_dicts = self.model_fn(history, context)
            
            if operator_dicts and isinstance(operator_dicts, list):
                operators = []
                for op_dict in operator_dicts[:k]:
                    op_type = op_dict.pop('type', None)
                    if op_type:
                        operators.append(Operator(type=op_type, params=op_dict))
                
                if operators:
                    return operators
        
        except Exception as e:
            print(f"Model proposer failed: {e}, falling back to adaptive")
        
        # Fallback
        if self.fallback_proposer:
            return self.fallback_proposer.propose_operators(
                problem, current_solution, history, trust_region, k
            )
        
        raise RuntimeError("Model proposer failed and no fallback available")


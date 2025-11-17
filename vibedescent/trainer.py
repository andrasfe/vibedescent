"""
Main Vibe Descent training loop.
"""

from typing import List, Tuple, Optional, Callable
import time
from dataclasses import dataclass, field

from .core import (
    Solution, Problem, Evaluator, Proposer, Critic, 
    Candidate, Operator, EvalResult
)
from .optimizer import OptimizerState, TrustRegion
from .config import ObjectiveConfig


@dataclass
class TrainingStats:
    """Statistics collected during training."""
    
    iterations: int = 0
    total_evaluations: int = 0
    total_time_ms: float = 0.0
    improvements: int = 0
    
    best_loss: float = float('inf')
    best_objective: float = 0.0
    best_gap: float = float('inf')
    
    history: List[Candidate] = field(default_factory=list)
    
    def update(self, state: OptimizerState, elapsed_ms: float):
        """Update stats from optimizer state."""
        self.iterations = state.iterations
        self.total_evaluations = state.total_evaluations
        self.total_time_ms += elapsed_ms
        self.improvements = state.improvements
        self.best_loss = state.best_loss
        self.best_objective = state.best_objective
        self.best_gap = state.best_gap
    
    def to_dict(self):
        """Convert to dictionary for logging."""
        return {
            'iterations': self.iterations,
            'total_evaluations': self.total_evaluations,
            'total_time_s': self.total_time_ms / 1000.0,
            'improvements': self.improvements,
            'best_loss': self.best_loss,
            'best_objective': self.best_objective,
            'best_gap': self.best_gap,
        }


class VibeDescentTrainer:
    """
    Main trainer implementing the Vibe Descent algorithm.
    
    Orchestrates the proposal → evaluation → selection → update loop
    with trust region management and stopping criteria.
    """
    
    def __init__(
        self,
        problem: Problem,
        evaluator: Evaluator,
        proposer: Proposer,
        critic: Critic,
        config: ObjectiveConfig,
        initial_solution: Optional[Solution] = None,
        verbose: bool = True,
        log_callback: Optional[Callable[[int, Candidate, OptimizerState], None]] = None
    ):
        """
        Initialize Vibe Descent trainer.
        
        Args:
            problem: Problem instance to optimize
            evaluator: Evaluator for checking solutions
            proposer: Proposer for generating operators
            critic: Critic for selecting candidates
            config: Objective configuration
            initial_solution: Starting solution (if None, must provide in train())
            verbose: Whether to print progress
            log_callback: Optional callback for custom logging
        """
        self.problem = problem
        self.evaluator = evaluator
        self.proposer = proposer
        self.critic = critic
        self.config = config
        self.initial_solution = initial_solution
        self.verbose = verbose
        self.log_callback = log_callback
        
        # Initialize optimizer state
        trust_region_config = config.trust_region
        self.optimizer_state = OptimizerState(
            trust_region=TrustRegion(
                min_size=trust_region_config.get('min', 0.5),
                max_size=trust_region_config.get('max', 2.0),
                initial_size=trust_region_config.get('init', 1.0),
                expand_factor=trust_region_config.get('expand', 1.2),
                shrink_factor=trust_region_config.get('shrink', 0.8),
            ),
            patience_max=config.stop.get('patience', 3)
        )
        self.optimizer_state.patience_left = self.optimizer_state.patience_max
        
        # Training state
        self.best_solution: Optional[Solution] = None
        self.best_evaluation: Optional[EvalResult] = None
        self.history: List[Candidate] = []
        self.stats = TrainingStats()
    
    def train(
        self,
        initial_solution: Optional[Solution] = None,
        max_iterations: Optional[int] = None,
        candidates_per_iteration: int = 6,
    ) -> Tuple[Solution, EvalResult, TrainingStats]:
        """
        Run the Vibe Descent training loop.
        
        Args:
            initial_solution: Starting solution (overrides constructor)
            max_iterations: Maximum iterations (overrides config)
            candidates_per_iteration: Number of operators to try per iteration
        
        Returns:
            Tuple of (best_solution, best_evaluation, stats)
        """
        # Setup
        start_time = time.time()
        current_solution = initial_solution or self.initial_solution
        if current_solution is None:
            raise ValueError("Must provide initial_solution")
        
        max_iters = max_iterations or self.config.stop.get('max_iters', 30)
        min_iters = self.config.stop.get('min_iters', 1)
        target_gap = self.config.stop.get('target_gap_pct', None)
        
        # Evaluate initial solution
        init_eval = self.evaluator.evaluate(self.problem, current_solution)
        init_eval.gap_pct = init_eval.compute_gap()
        init_loss = self.evaluator.compute_loss(init_eval, self.config.weights)
        
        self.best_solution = current_solution.copy()
        self.best_evaluation = init_eval
        self.optimizer_state.best_loss = init_loss
        self.optimizer_state.best_objective = init_eval.objective_value
        self.optimizer_state.best_gap = init_eval.gap_pct
        
        if self.verbose:
            self._print_header()
            self._print_initial(init_eval)
        
        # Main loop
        for iteration in range(1, max_iters + 1):
            iter_start = time.time()
            self.optimizer_state.iterations = iteration
            
            # 1) Propose operators
            operators = self.proposer.propose_operators(
                problem=self.problem,
                current_solution=self.best_solution,
                history=self.history,
                trust_region=self.optimizer_state.trust_region.current_size,
                k=candidates_per_iteration
            )
            
            # 2) Evaluate candidates
            candidates = self._evaluate_candidates(operators, self.best_solution)
            self.optimizer_state.total_evaluations += len(candidates)
            
            # 3) Select best candidate
            best_candidate = self.critic.select_best(
                candidates,
                context=self.problem.get_context()
            )
            
            # 4) Update state
            improved = self._update_state(best_candidate)
            
            # 5) Record history
            self.history.append(best_candidate)
            
            # 6) Log progress
            iter_time = (time.time() - iter_start) * 1000
            self.stats.update(self.optimizer_state, iter_time)
            self.stats.history.append(best_candidate)
            
            if self.verbose:
                self._print_iteration(iteration, best_candidate, improved)
            
            if self.log_callback:
                self.log_callback(iteration, best_candidate, self.optimizer_state)
            
            # 7) Check stopping criteria
            if self.optimizer_state.should_stop(target_gap, max_iters, min_iters):
                stop_reason = self._get_stop_reason(target_gap, max_iters)
                if self.verbose:
                    print(f"\nStopping: {stop_reason}")
                break
        
        # Final stats
        total_time = time.time() - start_time
        self.stats.total_time_ms = total_time * 1000
        
        if self.verbose:
            self._print_summary()
        
        return self.best_solution, self.best_evaluation, self.stats
    
    def _evaluate_candidates(
        self, 
        operators: List[Operator], 
        base_solution: Solution
    ) -> List[Candidate]:
        """Evaluate all operator applications."""
        candidates = []
        
        for op in operators:
            try:
                # Apply operator to get new solution
                new_solution = self._apply_operator(op, base_solution)
                
                # Evaluate
                t0 = time.time()
                evaluation = self.evaluator.evaluate(self.problem, new_solution)
                evaluation.runtime_ms = (time.time() - t0) * 1000
                evaluation.gap_pct = evaluation.compute_gap()
                
                # Compute loss
                loss = self.evaluator.compute_loss(evaluation, self.config.weights)
                
                candidates.append(Candidate(
                    operator=op,
                    solution=new_solution,
                    evaluation=evaluation,
                    loss=loss
                ))
            
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Operator {op} failed: {e}")
                continue
        
        if not candidates:
            raise RuntimeError("No valid candidates generated")
        
        return candidates
    
    def _apply_operator(self, operator: Operator, solution: Solution) -> Solution:
        """
        Apply an operator to a solution.
        
        This is a hook that should be overridden or configured with
        problem-specific operator application logic.
        """
        # This needs to be provided by the problem-specific setup
        # For now, just return a copy
        return solution.copy()
    
    def _update_state(self, candidate: Candidate) -> bool:
        """
        Update optimizer state based on selected candidate.
        
        Returns True if this was an improvement.
        """
        improved = False
        
        if candidate.evaluation.feasible:
            improved = (
                candidate.loss < self.optimizer_state.best_loss
                or candidate.evaluation.gap_pct < self.optimizer_state.best_gap
            )
        
        if improved:
            self.best_solution = candidate.solution
            self.best_evaluation = candidate.evaluation
            
            self.optimizer_state.update_on_improvement(
                candidate.loss,
                candidate.evaluation.objective_value,
                candidate.evaluation.gap_pct
            )
        else:
            self.optimizer_state.update_on_stagnation()
        
        return improved
    
    def _get_stop_reason(self, target_gap: Optional[float], max_iters: int) -> str:
        """Determine why optimization stopped."""
        if self.optimizer_state.patience_left <= 0:
            return "patience exhausted"
        if target_gap and self.optimizer_state.best_gap <= target_gap:
            return "target gap reached"
        if self.optimizer_state.iterations >= max_iters:
            return "max iterations reached"
        return "unknown"
    
    # Logging helpers
    
    def _print_header(self):
        """Print header for progress output."""
        print("\n" + "="*80)
        print("Vibe Descent Optimization")
        print("="*80)
        print(f"Problem: {self.problem.name}")
        print(f"Max iterations: {self.config.stop.get('max_iters', 'unlimited')}")
        print(f"Target gap: {self.config.stop.get('target_gap_pct', 'none')}%")
        print(f"Patience: {self.config.stop.get('patience', 'unlimited')}")
        print("="*80)
    
    def _print_initial(self, evaluation: EvalResult):
        """Print initial solution info."""
        print(f"\nInitial solution:")
        print(f"  Objective: {evaluation.objective_value:.2f}")
        print(f"  Gap: {evaluation.gap_pct:.2f}%")
        print(f"  Feasible: {evaluation.feasible}")
        if evaluation.upper_bound:
            print(f"  Upper bound: {evaluation.upper_bound:.2f}")
        print()
    
    def _print_iteration(self, iteration: int, candidate: Candidate, improved: bool):
        """Print iteration progress."""
        marker = "✓" if improved else "·"
        gap_str = f"{candidate.evaluation.gap_pct:.2f}%" if candidate.evaluation.gap_pct < float('inf') else "inf"
        
        print(
            f"{marker} Iter {iteration:3d}: "
            f"obj={candidate.evaluation.objective_value:8.1f} | "
            f"gap={gap_str:>7} | "
            f"loss={candidate.loss:7.4f} | "
            f"op={str(candidate.operator):30s} | "
            f"trust={self.optimizer_state.trust_region.current_size:.2f} | "
            f"patience={self.optimizer_state.patience_left}"
        )
    
    def _print_summary(self):
        """Print final summary."""
        print("\n" + "="*80)
        print("Optimization Complete")
        print("="*80)
        print(f"Total iterations: {self.stats.iterations}")
        print(f"Total evaluations: {self.stats.total_evaluations}")
        print(f"Total time: {self.stats.total_time_ms/1000:.2f}s")
        print(f"Improvements: {self.stats.improvements}")
        print(f"\nBest solution:")
        print(f"  Objective: {self.best_evaluation.objective_value:.2f}")
        print(f"  Gap: {self.best_evaluation.gap_pct:.2f}%")
        print(f"  Loss: {self.optimizer_state.best_loss:.4f}")
        print(f"  Feasible: {self.best_evaluation.feasible}")
        print("="*80 + "\n")


# Vibe Descent

**A trust-region optimization loop for discrete problems**

Vibe Descent treats code or algorithm optimization like training a machine learning model:

- The **problem state** (code, configuration, solver strategy) acts as the model parameters.
- **Strategies / operators** play the role of gradient steps.
- **Evaluations** provide the loss signal (runtime, memory, gaps, correctness).
- The framework iteratively proposes new strategies, tests them, and adapts the search.

This document explains how the loop works, what inputs it expects, how loss is computed, and shows real iteration data from the Falling Squares benchmark.

---

## Architecture at a Glance

| Component | Role |
|-----------|------|
| **Problem & Solution** | Define instance data plus a mutable solution representation. |
| **Strategy Registry** | Lists runnable algorithms/approaches (e.g., `bitset`, `segment_tree`). |
| **Proposer** | Chooses which strategies or parameter tweaks to try each iteration. |
| **Evaluator** | Runs the candidate strategy, checks hard constraints, measures metrics. |
| **Critic** | Picks the best candidate (lowest loss) per iteration. |
| **Optimizer State** | Tracks trust region size, patience, and the best metrics so far. |
| **Trainer** | Executes the Vibe Descent loop end‑to‑end. |

**Key capabilities**

- **Strategy-aware inputs**: provide many alternative algorithms via the registry.
- **Trust region & patience**: constrains how big a change each iteration can make.
- **Multi-term loss**: combine runtime, memory, allocations, and correctness gaps.
- **Logging**: every iteration records the chosen strategy, measurements, and loss.

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vibedescent.git
cd vibedescent

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Quick Start

```bash
# Install
pip install -r requirements.txt
pip install -e .

# Run Falling Squares with strategies + logging
python examples/falling_squares/run.py \
  --random-cases 10 --case-length 600 \
  --iters 18 --k 10 --target-gap 0.02 --patience 5
```

All runs create JSONL logs under `examples/<example>/logs/<example>/`.

## Inputs & Iterative Flow

### 1. Strategy Layer

- Strategies encapsulate how to solve the problem (e.g. `bitset`, `diff_array`, `segment_tree`, `bucket_grid`, `block_dp`).
- Each strategy registers itself via `StrategyRegistry` and exposes optional tuning parameters (e.g., `bucket_size`, `block_size`).
- The proposer samples from this pool and can also apply parameter tweaks using `adjust_param` operators.

### 2. Iteration (Epoch) Loop

1. **Propose**: sample `k` strategies/parameter tweaks. Exploration rate introduces randomness; momentum biases toward recent winners.
2. **Evaluate**: For each candidate, run the solver, collect metrics (runtime, memory, allocations, gap, feasibility). All correctness constraints must pass.
3. **Select**: Critic picks the lowest-loss candidate (the “winner” of the iteration).
4. **Update**: If loss improves, trust region expands and patience resets; otherwise trust region shrinks and patience decreases.
5. **Stop** when:
   - Min iterations satisfied and target loss/gap reached, or
   - Patience exhausted, or
   - Max iterations reached.

Logs capture each iteration: chosen strategy, runtime, loss, trust region, patience, and the running best metrics.

### 3. Loss Function

Loss is a weighted sum of normalized metrics:

```
Loss = w_gap * (gap_pct / 100)
     + w_runtime * (runtime_ms / runtime_target)
     + w_memory * (memory_mb / memory_target)
     + w_allocations * (alloc_ratio)
     + w_violation * (#violations)
```

- Hard constraints (e.g., correctness) must be satisfied or the candidate is rejected.
- Targets/normalizers come from the `ObjectiveConfig.targets` section.
- Weights are user-defined; for Falling Squares we used `gap=0.15`, `runtime=0.45`, `memory=0.25`, `allocations=0.10`, `violation=0.05`.

---

## Example: Falling Squares

The [Falling Squares](https://leetcode.com/problems/falling-squares/) example demonstrates a diverse strategy pool and an objective that balances runtime, memory, and correctness.

### Available Strategies

- `naive` (O(n²) stack simulation)
- `segment_tree` / `segment_tree_lazy`
- `interval_map` (non-overlapping intervals)
- `diff_array`
- `bucket_grid` (with tunable `bucket_size`)
- `bitset` (dense array for small spans)
- `fenwick` (BIT)
- `block_dp` (blocked coordinate array, tunable `block_size`)
- `sparse_tree`

### Clean Run (seed 42, random cases 10, case length 600)

Command:
```
python examples/falling_squares/run.py \
  --random-cases 10 --case-length 600 \
  --iters 18 --k 10 --target-gap 0.02 --patience 5 --seed 42
```

Iterations:

| Iter | Operator | Strategy | Avg runtime (ms) | Loss | Trust | Notes |
|------|----------|----------|------------------|------|-------|-------|
| 0 | — | `segment_tree_lazy` (initial) | 8.08 | — | 1.0 | Feasible baseline |
| 1 | `set_strategy(diff_array)` | `diff_array` | 2.18 | 0.0239 | 1.20 | Large runtime drop |
| 2 | `set_strategy(bitset)` | `bitset` | 0.93 | 0.0180 | 1.44 | Further improvement |
| 3 | `adjust_param(block_size +22)` | `block_dp` | 0.83 | 0.0174 | 1.73 | Tests block tuning |
| 4 | `adjust_param(block_size -25)` | `block_dp` | 0.81 | 0.0173 | 2.00 | Best loss, stops |

Final result: `bitset` strategy, avg runtime ≈ **9.87 ms**, gap 0%, loss 0.0173. Complete logs reside in `examples/falling_squares/logs/falling_squares/`.

You can rerun the loop after clearing logs to obtain a fresh trace:

```bash
rm -rf examples/falling_squares/logs
python examples/falling_squares/run.py ... (same command as above)
```

### Example: VRPTW (Vehicle Routing with Time Windows)

The VRPTW example (`examples/vrptw/`) showcases how Vibe Descent can orchestrate multiple routing heuristics:

- **Strategies**: nearest-time-window greedy, Clark-Wright savings, randomized insertion, two-phase hybrid.
- **Evaluation**: enforces serving every customer within capacity/time windows, and measures total distance plus lateness penalties.
- **Loss**: balances route distance (runtime proxy) with lateness/missing-customer penalties (violations).

Run it with:

```bash
python examples/vrptw/run.py --customers 30 --capacity 40 --iters 14 --k 6
```

The log directory `examples/vrptw/logs/vrptw/` stores per-iteration measurements so you can inspect which heuristic configuration won.

Sample run (seed 3, 20 customers, capacity 35):

| Iter | Operator | Routes | Objective | Notes |
|------|----------|--------|-----------|-------|
| 0 | — (initial) | 20 singleton routes | 1501.6 | Greedy seed |
| 1 | `savings` | 10 routes | 1963.7 | Merges customers via savings |
| 2 | `nearest_time_window` | 6 routes | 846.3 | Major improvement |
| 3 | `two_phase` | 6 routes | 3604.1 | Hybrid tweak (worse) |
| 4 | `savings` | 6 routes | 1963.7 | Recovery |
| 5 | `savings` | 6 routes | 1963.7 | No change |
| 6 | `two_phase` | 6 routes | 1938.5 | Minor adjustment |
| 7 | `two_phase` | 6 routes | 2815.4 | Regression |
| 8 | `nearest_time_window` | 6 routes | 846.3 | Best objective revisited |

The best configuration in this run is the `nearest_time_window` strategy with 6 routes covering all customers and the lowest measured objective (distance + lateness penalties).

## Configuration (Objective File)

## Extending the Framework

1. **Define Solution & Problem** (`vibedescent.core`).
2. **Register strategies** either by implementing operators or plugging in solver functions via `StrategyRegistry`.
3. **Implement Evaluator** to enforce constraints and compute metrics.
4. **Configure Objective** (weights, targets, trust region, min_iters, patience).
5. **Run Trainer** with an initial solution and a proposer (Adaptive or Model‑based).

## Documentation & Support

- [Examples](examples/)
- [Strategy registry usage](examples/falling_squares/solvers.py)
- [Objective configs](docs/)
- Issues/PRs welcome!

---

### Citation

If you use Vibe Descent in academic work, please cite:

```bibtex
@software{ferenczi_vibedescent,
  author = {Andras L. Ferenczi},
  orcid = {0000-0001-6785-9416},
  title = {Vibe Descent: Trust-region optimization for agentic coding},
  year = {2025},
  url = {https://github.com/andrasfe/vibedescent}
}
```


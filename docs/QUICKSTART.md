# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Running Your First Optimization

### Example 1: Knapsack with Default Settings

```bash
cd examples/knapsack
python run.py
```

This will:
- Generate a random knapsack instance with 100 items
- Run Vibe Descent for up to 30 iterations
- Stop when gap ≤ 0.5% or patience exhausted
- Print progress and final results

**Expected output:**
```
================================================================================
Vibe Descent Optimization
================================================================================
Problem: random_n100
Max iterations: 30
Target gap: 0.5%
Patience: 3
================================================================================

Initial solution:
  Objective: 1523.00
  Gap: 8.45%
  Feasible: True
  Upper bound: 1664.30

✓ Iter   1: obj=  1598.0 | gap=  3.98% | loss=0.0478 | op=two_opt_swap(tries=150) | trust=1.20 | patience=3
✓ Iter   2: obj=  1612.0 | gap=  3.14% | loss=0.0394 | op=ruin_recreate(...) | trust=1.44 | patience=3
...
Stopping: target gap reached

================================================================================
Optimization Complete
================================================================================
Best objective: 1656.00
Gap: 0.49%
```

### Example 2: Small Instance (Exact Solver)

```bash
python run.py --n 40
```

For n ≤ 46, the meet-in-the-middle exact solver will be used, finding the true optimum.

### Example 3: Large Instance

```bash
python run.py --n 200 --iters 40 --k 8
```

- 200 items
- 40 iterations max
- 8 candidates per iteration

### Example 4: Custom Target

```bash
python run.py --target-gap 1.0 --patience 5 --iters 50
```

- Stop when gap ≤ 1.0%
- Allow 5 iterations without improvement
- Run up to 50 iterations

## Understanding the Output

### Progress Indicators

- **✓** = Improvement found (better objective or gap)
- **·** = No improvement (stagnation)

### Metrics

- **obj**: Objective value (total value of items)
- **gap**: Optimality gap vs fractional upper bound (%)
- **loss**: Scalar loss combining gap, runtime, etc.
- **op**: Operator that was applied
- **trust**: Current trust region size
- **patience**: Iterations left before early stop

### Trust Region Behavior

- Expands after improvements: `1.0 → 1.2 → 1.44 → ...`
- Shrinks after stagnation: `1.0 → 0.8 → 0.64 → ...`
- Influences operator parameters (e.g., swap tries, destroy fraction)

## Next Steps

1. **Modify parameters** to see how they affect optimization
2. **Try different problem sizes** (--n)
3. **Experiment with stopping criteria** (--target-gap, --patience)
4. **Look at the code** in `examples/knapsack/` to understand the implementation
5. **Create your own problem** following the patterns in the knapsack example

## Common Patterns

### Quick Test Run
```bash
python run.py --n 50 --iters 10 --quiet
```

### Thorough Optimization
```bash
python run.py --n 150 --iters 50 --k 8 --target-gap 0.3 --patience 5
```

### Benchmarking
```bash
for n in 50 100 150 200; do
    python run.py --n $n --iters 30 --quiet
done
```

## Troubleshooting

### "No valid candidates generated"
- Increase `--k` (candidates per iteration)
- Check operator implementations for bugs

### "Stagnation after few iterations"
- Increase `--patience`
- Try different initial solutions
- Add more operator diversity

### "Too slow"
- Reduce `--k`
- Reduce `--iters`
- Use `--quiet` to suppress output
- For large n, disable exact DP computation

## Configuration Files

You can also use YAML configuration files:

```bash
python run.py --config my_config.yaml
```

Example `my_config.yaml`:
```yaml
hard_constraints:
  - feasible

weights:
  gap: 0.70
  runtime: 0.20
  memory: 0.05
  violation: 0.05

trust_region:
  min: 0.5
  max: 2.0
  init: 1.0
  expand: 1.2
  shrink: 0.8

stop:
  target_gap_pct: 0.5
  patience: 3
  max_iters: 30
```

## What's Next?

- Read the [full documentation](../README.md)
- Explore the [API reference](api.md)
- Learn about [implementing new problems](tutorial.md)
- Check out the [concepts guide](concepts.md)


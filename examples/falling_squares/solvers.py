"""Solver implementations for the Falling Squares problem."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Dict, Any, Optional, Callable
from bisect import bisect_left
from collections import defaultdict
from vibedescent.strategy import Strategy, StrategyRegistry


Positions = Sequence[Sequence[int]]


def _stats_dict(memory_bytes: int, allocations: int) -> Dict[str, float]:
    return {
        "memory_bytes": float(memory_bytes),
        "allocations": float(allocations),
    }


def falling_squares_naive(
    positions: Positions,
    return_stats: bool = False,
) -> List[int] | Tuple[List[int], Dict[str, float]]:
    """Brute-force O(n^2) solution."""

    stacks: List[Tuple[int, int, int]] = []
    ans: List[int] = []
    max_height = 0

    for left, size in positions:
        right = left + size
        base = 0
        for l2, r2, h2 in stacks:
            if not (right <= l2 or r2 <= left):
                base = max(base, h2)
        new_height = base + size
        stacks.append((left, right, new_height))
        max_height = max(max_height, new_height)
        ans.append(max_height)
    stats = _stats_dict(memory_bytes=len(positions) * 3 * 8, allocations=len(positions))
    if return_stats:
        return ans, stats
    return ans


class SegmentTree:
    """Segment tree with range-assign updates and max queries."""

    __slots__ = ("n", "tree", "lazy")

    def __init__(self, n: int) -> None:
        self.n = n
        self.tree = [0] * (4 * n)
        self.lazy = [0] * (4 * n)

    def _push(self, node: int) -> None:
        if self.lazy[node] != 0:
            val = self.lazy[node]
            left = node * 2
            right = left + 1
            self.tree[left] = val
            self.tree[right] = val
            self.lazy[left] = val
            self.lazy[right] = val
            self.lazy[node] = 0

    def update(self, node: int, l: int, r: int, ql: int, qr: int, val: int) -> None:
        if ql > r or qr < l:
            return
        if ql <= l and r <= qr:
            self.tree[node] = val
            self.lazy[node] = val
            return
        self._push(node)
        mid = (l + r) // 2
        self.update(node * 2, l, mid, ql, qr, val)
        self.update(node * 2 + 1, mid + 1, r, ql, qr, val)
        self.tree[node] = max(self.tree[node * 2], self.tree[node * 2 + 1])

    def query(self, node: int, l: int, r: int, ql: int, qr: int) -> int:
        if ql > r or qr < l:
            return 0
        if ql <= l and r <= qr:
            return self.tree[node]
        self._push(node)
        mid = (l + r) // 2
        return max(
            self.query(node * 2, l, mid, ql, qr),
            self.query(node * 2 + 1, mid + 1, r, ql, qr),
        )


def _compress_coordinates(positions: Positions) -> Tuple[Dict[int, int], List[int]]:
    coords = set()
    for left, size in positions:
        coords.add(left)
        coords.add(left + size)
    sorted_coords = sorted(coords)
    mapping = {x: i for i, x in enumerate(sorted_coords)}
    return mapping, sorted_coords


def falling_squares_segment_tree(
    positions: Positions,
    params: Optional[Dict[str, Any]] = None,
    return_stats: bool = False,
) -> List[int] | Tuple[List[int], Dict[str, float]]:
    """Segment tree solution with coordinate compression."""

    params = params or {}
    mapping, coords = _compress_coordinates(positions)
    tree = SegmentTree(len(coords))
    ans: List[int] = []
    max_height = 0

    for left, size in positions:
        right = left + size
        l_idx = mapping[left]
        r_idx = mapping[right] - 1
        if r_idx < l_idx:
            r_idx = l_idx
        current = tree.query(1, 0, tree.n - 1, l_idx, r_idx)
        new_height = current + size
        tree.update(1, 0, tree.n - 1, l_idx, r_idx, new_height)
        max_height = max(max_height, new_height)
        ans.append(max_height)
    tree_nodes = 4 * len(coords)
    memory_bytes = (tree_nodes * 2 + len(coords)) * 8
    allocations = tree_nodes * 2 + len(coords)
    stats = _stats_dict(memory_bytes=memory_bytes, allocations=allocations)
    if return_stats:
        return ans, stats
    return ans


def falling_squares_interval_map(
    positions: Positions,
    return_stats: bool = False,
) -> List[int] | Tuple[List[int], Dict[str, float]]:
    """
    Interval map solution that maintains a set of non-overlapping
    intervals with constant heights.

    Complexity is O(k log k) where k is number of intervals (worst-case O(n^2)),
    but typically performs differently from the segment tree and naive approaches.
    """

    intervals: List[Tuple[int, int, int]] = []
    ans: List[int] = []
    max_height = 0

    for left, size in positions:
        right = left + size
        new_intervals: List[Tuple[int, int, int]] = []
        base = 0

        for l, r, h in intervals:
            if r <= left or l >= right:
                new_intervals.append((l, r, h))
            else:
                base = max(base, h)
                if l < left:
                    new_intervals.append((l, left, h))
                if r > right:
                    new_intervals.append((right, r, h))

        new_height = base + size
        new_intervals.append((left, right, new_height))

        # Merge adjacent intervals with identical heights
        new_intervals.sort()
        merged: List[Tuple[int, int, int]] = []
        for l, r, h in new_intervals:
            if not merged:
                merged.append((l, r, h))
                continue
            pl, pr, ph = merged[-1]
            if pr == l and ph == h:
                merged[-1] = (pl, r, ph)
            else:
                merged.append((l, r, h))

        intervals = merged
        max_height = max(max_height, new_height)
        ans.append(max_height)

    memory_bytes = len(intervals) * 3 * 8
    stats = _stats_dict(memory_bytes=memory_bytes, allocations=len(intervals))
    if return_stats:
        return ans, stats
    return ans


def falling_squares_diff_array(
    positions: Positions,
    return_stats: bool = False,
) -> List[int] | Tuple[List[int], Dict[str, float]]:
    """
    Coordinate-compressed difference array solution.

    Maintains explicit heights for each compressed interval.
    """
    mapping, coords = _compress_coordinates(positions)
    n = len(coords)
    heights = [0] * n
    ans: List[int] = []
    max_height = 0

    for left, size in positions:
        right = left + size
        l_idx = mapping[left]
        r_idx = mapping[right]
        base = 0
        for i in range(l_idx, r_idx):
            base = max(base, heights[i])
        new_height = base + size
        for i in range(l_idx, r_idx):
            heights[i] = new_height
        max_height = max(max_height, new_height)
        ans.append(max_height)

    memory_bytes = len(heights) * 8
    stats = _stats_dict(memory_bytes=memory_bytes, allocations=len(heights))
    if return_stats:
        return ans, stats
    return ans


def falling_squares_bucket_grid(
    positions: Positions,
    params: Optional[Dict[str, Any]] = None,
    return_stats: bool = False,
) -> List[int] | Tuple[List[int], Dict[str, float]]:
    """
    Bucketed interval storage that approximates interactions using coarse buckets.
    """
    params = params or {}
    bucket_size = params.get("bucket_size", 64)
    buckets: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    ans: List[int] = []
    max_height = 0

    for left, size in positions:
        right = left + size
        b_start = left // bucket_size
        b_end = (right - 1) // bucket_size

        base = 0
        for b in range(b_start, b_end + 1):
            for l, r, h in buckets[b]:
                if not (right <= l or r <= left):
                    base = max(base, h)

        new_height = base + size
        for b in range(b_start, b_end + 1):
            buckets[b].append((left, right, new_height))

        max_height = max(max_height, new_height)
        ans.append(max_height)

    total_intervals = sum(len(v) for v in buckets.values())
    memory_bytes = total_intervals * 3 * 8
    stats = _stats_dict(memory_bytes=memory_bytes, allocations=total_intervals)
    if return_stats:
        return ans, stats
    return ans


def falling_squares_bucket_grid_large(
    positions: Positions,
    params: Optional[Dict[str, Any]] = None,
    return_stats: bool = False,
) -> List[int] | Tuple[List[int], Dict[str, float]]:
    local_params = dict(params or {})
    local_params.setdefault("bucket_size", 128)
    return falling_squares_bucket_grid(positions, params=local_params, return_stats=return_stats)


def falling_squares_bitset(
    positions: Positions,
    params: Optional[Dict[str, Any]] = None,
    return_stats: bool = False,
) -> List[int] | Tuple[List[int], Dict[str, float]]:
    """
    Bitset-based method for small coordinate ranges.

    Uses a dense array of heights, so only suitable when coordinates span
    a relatively small domain. Provides a contrasting memory/runtime profile.
    """
    params = params or {}
    span_limit = params.get("span_limit", 6000)

    if not positions:
        stats = _stats_dict(memory_bytes=0, allocations=0)
        return ([], stats) if return_stats else []

    min_coord = min(p[0] for p in positions)
    max_coord = max(p[0] + p[1] for p in positions)
    span = max_coord - min_coord + 1

    # Guard against huge allocations; fall back to diff array
    if span > span_limit:
        return falling_squares_diff_array(positions, return_stats=return_stats)

    heights = [0] * span
    ans: List[int] = []
    max_height = 0

    for left, size in positions:
        offset_left = left - min_coord
        offset_right = offset_left + size
        base = max(heights[offset_left:offset_right])
        new_height = base + size
        for i in range(offset_left, offset_right):
            heights[i] = new_height
        max_height = max(max_height, new_height)
        ans.append(max_height)

    memory_bytes = len(heights) * 8
    stats = _stats_dict(memory_bytes=memory_bytes, allocations=len(heights))
    if return_stats:
        return ans, stats
    return ans


def falling_squares_bitset_small_span(
    positions: Positions,
    params: Optional[Dict[str, Any]] = None,
    return_stats: bool = False,
) -> List[int] | Tuple[List[int], Dict[str, float]]:
    local_params = dict(params or {})
    local_params.setdefault("span_limit", 4000)
    return falling_squares_bitset(positions, params=local_params, return_stats=return_stats)


def falling_squares_bitset_large_span(
    positions: Positions,
    params: Optional[Dict[str, Any]] = None,
    return_stats: bool = False,
) -> List[int] | Tuple[List[int], Dict[str, float]]:
    local_params = dict(params or {})
    local_params.setdefault("span_limit", 12000)
    return falling_squares_bitset(positions, params=local_params, return_stats=return_stats)


def falling_squares_fenwick(
    positions: Positions,
    return_stats: bool = False,
) -> List[int] | Tuple[List[int], Dict[str, float]]:
    """
    Fenwick tree storing maximum heights. Provides an alternative
    tree-based solution with different constant factors than segment tree.
    """
    mapping, coords = _compress_coordinates(positions)
    n = len(coords) + 2
    fenwick = [0] * (n + 2)

    def update(idx: int, value: int):
        while idx < len(fenwick):
            fenwick[idx] = max(fenwick[idx], value)
            idx += idx & -idx

    def query(idx: int) -> int:
        result = 0
        while idx > 0:
            result = max(result, fenwick[idx])
            idx -= idx & -idx
        return result

    ans: List[int] = []
    max_height = 0

    for left, size in positions:
        right = left + size
        l_idx = mapping[left] + 1
        r_idx = mapping[right] + 1
        base = query(r_idx - 1)
        new_height = base + size
        update(r_idx - 1, new_height)
        max_height = max(max_height, new_height)
        ans.append(max_height)

    memory_bytes = len(fenwick) * 8
    stats = _stats_dict(memory_bytes=memory_bytes, allocations=len(fenwick))
    if return_stats:
        return ans, stats
    return ans


def falling_squares_block_dp(
    positions: Positions,
    params: Optional[Dict[str, Any]] = None,
    return_stats: bool = False,
) -> List[int] | Tuple[List[int], Dict[str, float]]:
    """
    Blocked coordinate array with precomputed block maxima.
    """
    params = params or {}
    block_size = max(8, int(params.get("block_size", 64)))

    mapping, coords = _compress_coordinates(positions)
    n = len(coords)
    heights = [0] * n
    block_count = (n + block_size - 1) // block_size
    block_max = [0] * block_count

    def range_max(l: int, r: int) -> int:
        res = 0
        while l <= r:
            if l % block_size == 0 and l + block_size - 1 <= r:
                res = max(res, block_max[l // block_size])
                l += block_size
            else:
                res = max(res, heights[l])
                l += 1
        return res

    def range_update(l: int, r: int, value: int):
        i = l
        while i <= r:
            heights[i] = value
            block_idx = i // block_size
            start = block_idx * block_size
            end = min(start + block_size, n)
            block_max[block_idx] = max(block_max[block_idx], max(heights[start:end]))
            i += 1

    ans: List[int] = []
    max_height = 0

    for left, size in positions:
        right = left + size
        l_idx = mapping[left]
        r_idx = mapping[right] - 1
        base = range_max(l_idx, r_idx)
        new_height = base + size
        range_update(l_idx, r_idx, new_height)
        max_height = max(max_height, new_height)
        ans.append(max_height)

    memory_bytes = (len(heights) + len(block_max)) * 8
    stats = _stats_dict(memory_bytes=memory_bytes, allocations=len(heights) + len(block_max))
    if return_stats:
        return ans, stats
    return ans


def falling_squares_block_dp_fine(
    positions: Positions,
    params: Optional[Dict[str, Any]] = None,
    return_stats: bool = False,
) -> List[int] | Tuple[List[int], Dict[str, float]]:
    local_params = dict(params or {})
    local_params.setdefault("block_size", 32)
    return falling_squares_block_dp(positions, params=local_params, return_stats=return_stats)


def falling_squares_sparse_tree(
    positions: Positions,
    return_stats: bool = False,
) -> List[int] | Tuple[List[int], Dict[str, float]]:
    """
    Sparse tree storing only visited leaves.
    """
    if not positions:
        stats = _stats_dict(memory_bytes=0, allocations=0)
        return ([], stats) if return_stats else []

    coords = sorted({c for left, size in positions for c in (left, left + size)})
    tree: Dict[int, int] = {}

    def update(l: int, r: int, h: int, idx: int = 1, tl: int = 0, tr: Optional[int] = None):
        if tr is None:
            tr = len(coords) - 1
        if l > r:
            return
        if l == tl and r == tr:
            tree[idx] = max(tree.get(idx, 0), h)
            return
        tm = (tl + tr) // 2
        update(l, min(r, tm), h, idx * 2, tl, tm)
        update(max(l, tm + 1), r, h, idx * 2 + 1, tm + 1, tr)

    def query(pos: int, idx: int = 1, tl: int = 0, tr: Optional[int] = None) -> int:
        if tr is None:
            tr = len(coords) - 1
        res = tree.get(idx, 0)
        if tl == tr:
            return res
        tm = (tl + tr) // 2
        if pos <= tm:
            return max(res, query(pos, idx * 2, tl, tm))
        return max(res, query(pos, idx * 2 + 1, tm + 1, tr))

    ans: List[int] = []
    max_height = 0
    mapping = {c: i for i, c in enumerate(coords)}

    for left, size in positions:
        right = left + size
        l_idx = mapping[left]
        r_idx = mapping[right] - 1
        base = query(r_idx)
        new_height = base + size
        update(l_idx, r_idx, new_height)
        max_height = max(max_height, new_height)
        ans.append(max_height)

    memory_bytes = len(tree) * 16
    stats = _stats_dict(memory_bytes=memory_bytes, allocations=len(tree))
    if return_stats:
        return ans, stats
    return ans


strategy_registry = StrategyRegistry()


def _register_strategy(
    name: str,
    fn: Callable[..., Any],
    *,
    supports_params: bool = False,
    description: str = "",
    metadata: Optional[Dict[str, Any]] = None,
):
    metadata = metadata or {}

    def runner(positions: Positions, params: Optional[Dict[str, Any]] = None, return_stats: bool = False):
        if supports_params:
            return fn(positions, params=params or {}, return_stats=return_stats)
        return fn(positions, return_stats=return_stats)

    strategy_registry.register(
        Strategy(
            name=name,
            runner=runner,
            description=description,
            supports_params=supports_params,
            metadata=metadata,
        )
    )


_register_strategy("naive", falling_squares_naive, description="O(n^2) baseline")
_register_strategy("segment_tree", falling_squares_segment_tree, supports_params=True, description="Segment tree with coordinate compression")
_register_strategy("segment_tree_lazy", falling_squares_segment_tree, supports_params=True, description="Segment tree variant (alias for lazy tuning)")
_register_strategy("interval_map", falling_squares_interval_map, description="Non-overlapping interval maintenance")
_register_strategy("diff_array", falling_squares_diff_array, description="Difference-array on compressed coordinates")
_register_strategy("bucket_grid", falling_squares_bucket_grid, supports_params=True, description="Bucketed overlap checks")
_register_strategy(
    "bitset",
    falling_squares_bitset,
    supports_params=True,
    description="Dense bitset heights for small spans",
    metadata={"params": ["span_limit"]},
)
_register_strategy(
    "bitset_small_span",
    falling_squares_bitset_small_span,
    supports_params=True,
    description="Bitset tuned for smaller spans",
)
_register_strategy(
    "bitset_large_span",
    falling_squares_bitset_large_span,
    supports_params=True,
    description="Bitset tuned for larger spans",
)
_register_strategy("fenwick", falling_squares_fenwick, description="Fenwick tree (BIT) maximum heights")
_register_strategy("block_dp", falling_squares_block_dp, supports_params=True, description="Blocked coordinate DP array")
_register_strategy(
    "block_dp_fine",
    falling_squares_block_dp_fine,
    supports_params=True,
    description="Blocked coordinate DP array with finer blocks",
)
_register_strategy("sparse_tree", falling_squares_sparse_tree, description="Sparse tree storing visited leaves")
_register_strategy(
    "bucket_grid_large",
    falling_squares_bucket_grid_large,
    supports_params=True,
    description="Bucket grid with larger bucket size",
)


def run_solver(
    strategy: str,
    params: Dict[str, Any],
    positions: Positions,
    *,
    return_stats: bool = False,
) -> List[int] | Tuple[List[int], Dict[str, float]]:
    return strategy_registry.run(strategy.lower(), positions, params=params, return_stats=return_stats)


def list_strategies() -> List[str]:
    return strategy_registry.list_names()


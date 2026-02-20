#!/usr/bin/env python3
"""
Generate LRA ListOps-style data (TSV: Source, Target) without TensorFlow.
Matches the format expected by src/data/lra.py and the official LRA listops task.
Output: basic_train.tsv, basic_val.tsv, basic_test.tsv under output_dir.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

# LRA listops operators and values (same as official lra_benchmarks/data/listops.py)
MIN, MAX, MED, SUM_MOD = "[MIN", "[MAX", "[MED", "[SM"
END = "]"
OPERATORS = [MIN, MAX, MED, SUM_MOD]
VALUES = list(range(10))
VALUE_P = 0.25


def generate_tree(depth: int, max_depth: int, max_args: int) -> tuple[tuple | int, int]:
    """Generate tree-like equation. Returns (tree, length)."""
    r = random.random() if depth < max_depth else 1.0
    if r > VALUE_P:
        return random.choice(VALUES), 1
    length = 2
    num_values = random.randint(2, max_args)
    sub_trees = []
    for _ in range(num_values):
        sub_t, sub_l = generate_tree(depth + 1, max_depth, max_args)
        sub_trees.append(sub_t)
        length += sub_l
    op = random.choice(OPERATORS)
    t = (op, sub_trees[0])
    for v in sub_trees[1:]:
        t = (t, v)
    t = (t, END)
    return t, length


def to_string(t: tuple | int | str, parens: bool = True) -> str:
    if isinstance(t, str):
        return t
    if isinstance(t, int):
        return str(t)
    return "( " + to_string(t[0]) + " " + to_string(t[1]) + " )"


def to_value(t: tuple | int | str):
    """Compute the output of equation t (class 0-9). Returns int or (op, list) for unsaturated."""
    if not isinstance(t, tuple):
        return t
    left, right = to_value(t[0]), to_value(t[1])
    if left in OPERATORS:
        return (left, [right])
    if right == END:
        op, vals = left
        if op == MIN:
            return min(vals)
        if op == MAX:
            return max(vals)
        if op == MED:
            return int(sorted(vals)[len(vals) // 2])
        if op == SUM_MOD:
            return sum(vals) % 10
    if isinstance(left, tuple):
        return (left[0], left[1] + [right])
    return right


def main() -> None:
    p = argparse.ArgumentParser(description="Generate LRA ListOps TSV data")
    p.add_argument("--output_dir", type=Path, default=Path("output"),
                   help="Directory for basic_*.tsv")
    p.add_argument("--task", default="basic")
    p.add_argument("--num_train", type=int, default=96000)
    p.add_argument("--num_val", type=int, default=2000)
    p.add_argument("--num_test", type=int, default=2000)
    p.add_argument("--max_depth", type=int, default=10)
    p.add_argument("--max_args", type=int, default=10)
    p.add_argument("--max_length", type=int, default=2000)
    p.add_argument("--min_length", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total = args.num_train + args.num_val + args.num_test
    data = set()
    while len(data) < total:
        tree, length = generate_tree(1, args.max_depth, args.max_args)
        if args.min_length <= length <= args.max_length:
            data.add(tree)
        if len(data) % 5000 == 0 and len(data) > 0:
            print(f"Generated {len(data)}/{total} examples...")
    examples = [(to_string(ex), to_value(ex)) for ex in data]
    train = examples[: args.num_train]
    val = examples[args.num_train : args.num_train + args.num_val]
    test = examples[args.num_train + args.num_val :]

    for name, rows in [("train", train), ("val", val), ("test", test)]:
        path = args.output_dir / f"{args.task}_{name}.tsv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["Source", "Target"])
            w.writerows(rows)
        print(f"Wrote {path} ({len(rows)} rows)")
    print(f"Done. ListOps data in {args.output_dir}")


if __name__ == "__main__":
    main()

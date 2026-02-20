"""
Statistics over runs: mean, std, confidence intervals, significance tests.
Input: list of per-run metrics; output: summary and optional p-value.
"""

from typing import Any, Optional, Sequence, Union

import numpy as np


def mean_std(
    values: Sequence[Union[int, float]],
) -> tuple[float, float]:
    """
    Mean and standard deviation over a list of values (e.g. one metric across seeds).
    """
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0)


def confidence_interval_95(
    values: Sequence[Union[int, float]],
) -> tuple[float, float, float]:
    """
    95% confidence interval for the mean (assuming approximate normality).
    Uses t-distribution with n-1 degrees of freedom.

    Returns:
        (mean, lower_bound, upper_bound)
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = np.mean(arr)
    if n == 1:
        return float(mean), float(mean), float(mean)
    sem = np.std(arr, ddof=1) / np.sqrt(n)
    try:
        from scipy import stats
        t = stats.t.ppf(0.975, df=n - 1)
    except ImportError:
        # Fallback: use 1.96 for large n (z-interval)
        t = 1.96
    margin = t * sem
    return float(mean), float(mean - margin), float(mean + margin)


def summary_over_runs(
    values: Sequence[Union[int, float]],
    use_confidence_interval: bool = True,
) -> dict[str, float]:
    """
    Summary statistics over runs: mean, std, and optionally 95% CI.

    Returns:
        {"mean": float, "std": float}
        and if use_confidence_interval: "ci95_lower", "ci95_upper".
    """
    m, s = mean_std(values)
    out = {"mean": m, "std": s}
    if use_confidence_interval and len(values) > 1:
        _, lo, hi = confidence_interval_95(values)
        out["ci95_lower"] = lo
        out["ci95_upper"] = hi
    return out


def t_test_independent(
    group_a: Sequence[Union[int, float]],
    group_b: Sequence[Union[int, float]],
) -> tuple[float, float]:
    """
    Two-sample (unpaired) t-test: compare means of two independent groups.
    Returns (t_statistic, p_value). Use when you have two different methods
    (e.g. ION vs GRU) each run with several seeds.
    """
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), 1.0
    try:
        from scipy import stats
        t_stat, p_val = stats.ttest_ind(a, b)
        return float(t_stat), float(p_val)
    except ImportError:
        # No scipy: return t-statistic and placeholder p-value
        n1, n2 = len(a), len(b)
        m1, m2 = np.mean(a), np.mean(b)
        v1 = np.var(a, ddof=1) if n1 > 1 else 0.0
        v2 = np.var(b, ddof=1) if n2 > 1 else 0.0
        se = np.sqrt(v1 / n1 + v2 / n2)
        if se == 0:
            return 0.0, 1.0
        t = (m1 - m2) / se
        return float(t), float("nan")


def wilcoxon_signed_rank(
    group_a: Sequence[Union[int, float]],
    group_b: Sequence[Union[int, float]],
) -> tuple[float, float]:
    """
    Wilcoxon signed-rank test (paired). Use when the same seeds were used
    for both methods (e.g. ION and GRU both run with seeds [42, 123, 456]).
    Returns (statistic, p_value).
    """
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    if len(a) != len(b) or len(a) < 2:
        return float("nan"), 1.0
    try:
        from scipy import stats
        stat, p_val = stats.wilcoxon(a, b)
        return float(stat), float(p_val)
    except ImportError:
        return float("nan"), 1.0


def cohens_d(
    group_a: Sequence[Union[int, float]],
    group_b: Sequence[Union[int, float]],
) -> float:
    """
    Cohen's d effect size (pooled std): (mean_a - mean_b) / pooled_std.
    Returns 0 if pooled std is 0 or if either group has < 2 samples.
    """
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    n1, n2 = len(a), len(b)
    m1, m2 = np.mean(a), np.mean(b)
    v1 = np.var(a, ddof=1)
    v2 = np.var(b, ddof=1)
    pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    if pooled_var <= 0:
        return 0.0
    return float((m1 - m2) / np.sqrt(pooled_var))


def compare_two_methods(
    metrics_a: Sequence[Union[int, float]],
    metrics_b: Sequence[Union[int, float]],
    paired: bool = False,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Compare two methods (e.g. ION vs baseline) over runs.
    Returns summary for each and a significance test.

    Args:
        metrics_a: Per-run metric for method A (e.g. test accuracy for ION).
        metrics_b: Per-run metric for method B (e.g. test accuracy for GRU).
        paired: If True, use Wilcoxon signed-rank; else independent t-test.
        alpha: Significance level (e.g. 0.05).

    Returns:
        {
            "method_a": {"mean": ..., "std": ..., "ci95_lower": ..., "ci95_upper": ...},
            "method_b": {...},
            "test": "wilcoxon" | "ttest",
            "statistic": float,
            "p_value": float,
            "significant": bool (p_value < alpha),
        }
    """
    sum_a = summary_over_runs(metrics_a)
    sum_b = summary_over_runs(metrics_b)
    if paired:
        stat, p = wilcoxon_signed_rank(metrics_a, metrics_b)
        test_name = "wilcoxon"
    else:
        stat, p = t_test_independent(metrics_a, metrics_b)
        test_name = "ttest"
    d = cohens_d(metrics_a, metrics_b)
    return {
        "method_a": sum_a,
        "method_b": sum_b,
        "test": test_name,
        "statistic": stat,
        "p_value": p,
        "significant": p < alpha,
        "alpha": alpha,
        "cohens_d": d,
    }

"""Benchmark: scipy.linalg.solve(assume_a="pos") vs np.linalg.solve for FastL2LiR.

Section A — Single-RHS (n_units=1):
  Models the feature-selection ON path in __sub_fit: each output unit is solved
  independently with a (n_feat+1,) RHS vector under threadpool_limits(limits=1).
  Measures per-unit solve time.

Section B — Projected decoder run-time:
  Takes the per-unit times from Section A and projects to realistic decoder sizes
  (n_units = 1000, 10000, 100000) without actually running the loop.
  The feature-selection path is a pure sequential loop; the ratio does not change
  with n_units, so this projection is exact.

W0 is built as newX.T @ newX + alpha*I (same as __sub_fit), so the conditioning
is realistic (SPD, well-conditioned with alpha=100).

Run:
    python bench_solver.py              # 1 BLAS thread (default, matches decoding)
    python bench_solver.py --no-thread-limit  # use default BLAS thread count
"""

import argparse
import sys
import time

import numpy as np
from scipy import linalg as sp_linalg
from threadpoolctl import threadpool_info, threadpool_limits

from fastl2lir.fastl2lir import _solve_numpy, _solve_scipy

N_SAMPLES = 6000
ALPHA = 100.0
DTYPES = [np.float32, np.float64]
NFEAT_SWEEP = [50, 100, 200, 500, 1000, 2000]
N_UNITS_SWEEP = [1, 10, 100, 1000]
N_REPEATS = 10
WARMUP = 2
# Projected decoder sizes for Section B (no actual loop; just t_per_unit × n_units).
N_UNITS_PROJECTED = [1_000, 10_000, 100_000]


def make_normal_equation(
    n_feat: int,
    dtype=np.float32,
    n_samples: int = N_SAMPLES,
    alpha: float = ALPHA,
    seed: int = 0,
):
    """Build (W0, rhs) the way __sub_fit feature-selection path does for one unit.

    W0  : (n_feat+1, n_feat+1) SPD
    rhs : (n_feat+1,) — single output unit
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_feat)).astype(dtype)
    Xa = np.hstack([X, np.ones((n_samples, 1), dtype=dtype)])  # add bias column
    W0 = Xa.T @ Xa + alpha * np.eye(n_feat + 1, dtype=dtype)
    y = rng.standard_normal(n_samples).astype(dtype)
    rhs = Xa.T @ y  # (n_feat+1,)
    return W0, rhs


def bench_solve_once(solver_fn, W0, rhs, n_repeats: int = N_REPEATS, warmup: int = WARMUP):
    """Return median wall-time in seconds for a single solve call.

    Copies W0 and rhs on every call: LAPACK dpotrf/dgesv overwrite 'a' in-place,
    so without copying the factorisation from the first call corrupts later ones.
    """
    for _ in range(warmup):
        solver_fn(W0.copy(), rhs.copy())
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        solver_fn(W0.copy(), rhs.copy())
        times.append(time.perf_counter() - t0)
    return float(np.median(times))




def print_env():
    print(f"Python {sys.version}")
    print(f"NumPy {np.__version__}, SciPy {sp_linalg.__name__}")
    print(f"N_SAMPLES={N_SAMPLES}, ALPHA={ALPHA}, N_REPEATS={N_REPEATS}")
    infos = threadpool_info()
    if infos:
        for info in infos:
            print(
                f"BLAS: {info.get('user_api')} / {info.get('internal_api')} "
                f"{info.get('version')} (threads: {info.get('num_threads')})"
            )
    else:
        print("BLAS: (none detected by threadpoolctl, e.g. Accelerate)")


def section_a(limit_threads: bool) -> dict:
    """Run Section A and return {(dtype_name, n_feat): (t_scipy, t_numpy)}."""
    print("\n" + "=" * 60)
    print("Section A: Single-RHS (n_units=1), n_feat sweep")
    print("Per-unit solve time; ratio = scipy / numpy (>1 = scipy slower)")
    print("=" * 60)

    results = {}
    ctx = threadpool_limits(limits=1, user_api="blas") if limit_threads else _nullctx()
    with ctx:
        for dtype in DTYPES:
            dname = np.dtype(dtype).name
            print(f"\ndtype={dname}")
            hdr = f"  {'n_feat':>7} {'matrix':>9} {'scipy_us':>10} {'numpy_us':>10} {'ratio':>7}"
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for n_feat in NFEAT_SWEEP:
                W0, rhs = make_normal_equation(n_feat, dtype=dtype)
                dim = W0.shape[0]
                ts = bench_solve_once(_solve_scipy, W0, rhs)
                tn = bench_solve_once(_solve_numpy, W0, rhs)
                results[(dname, n_feat)] = (ts, tn)
                print(
                    f"  {n_feat:>7} {dim:>4}x{dim:<4} "
                    f"{ts * 1e6:>9.1f} {tn * 1e6:>9.1f} {ts / tn:>7.2f}"
                )
    return results


def section_b(per_unit: dict):
    """Section B: project per-unit times from Section A to full decoder sizes."""
    print("\n" + "=" * 60)
    print("Section B: Projected total decoder time (t_per_unit × n_units)")
    print("Feature-selection path is a pure sequential loop; ratio is constant.")
    print("ratio = scipy / numpy (>1 = scipy slower)")
    print("=" * 60)

    for dtype in DTYPES:
        dname = np.dtype(dtype).name
        print(f"\ndtype={dname}")

        # ratio grid
        units_hdr = "  n_units→  |" + "".join(f" {u:>10}" for u in N_UNITS_PROJECTED)
        print(f"\n  ratio grid (scipy/numpy)")
        print(units_hdr)
        print("  " + "-" * (len(units_hdr) - 2))
        for n_feat in NFEAT_SWEEP:
            ts, tn = per_unit[(dname, n_feat)]
            row = f"  {n_feat:>7} |"
            for n_units in N_UNITS_PROJECTED:
                row += f" {ts / tn:>10.2f}"
            print(row)

        # absolute time table
        print(f"\n  absolute projected time (seconds)")
        hdr2 = (
            f"  {'n_feat':>7} {'n_units':>10} "
            f"{'scipy_s':>10} {'numpy_s':>10} {'diff_s':>10} {'ratio':>7}"
        )
        print(hdr2)
        print("  " + "-" * (len(hdr2) - 2))
        for n_feat in NFEAT_SWEEP:
            ts, tn = per_unit[(dname, n_feat)]
            for n_units in N_UNITS_PROJECTED:
                print(
                    f"  {n_feat:>7} {n_units:>10} "
                    f"{ts * n_units:>10.1f} {tn * n_units:>10.1f} "
                    f"{(ts - tn) * n_units:>+10.1f} {ts / tn:>7.2f}"
                )


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-thread-limit",
        action="store_true",
        help="Do not restrict BLAS threads (default: limits=1, matches decoding)",
    )
    args = parser.parse_args()
    limit = not args.no_thread_limit

    print_env()
    if limit:
        print("\nRunning under threadpool_limits(limits=1, user_api='blas')")
    else:
        print("\nRunning with default BLAS thread count (--no-thread-limit)")

    per_unit = section_a(limit_threads=limit)
    section_b(per_unit)


if __name__ == "__main__":
    main()

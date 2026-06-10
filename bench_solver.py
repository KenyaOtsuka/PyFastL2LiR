"""Benchmark: scipy.linalg.solve(assume_a="pos") vs np.linalg.solve for FastL2LiR.

Section A — Single-RHS (n_units=1):
  Models the feature-selection ON path in __sub_fit: each output unit is solved
  independently with a (n_feat+1,) RHS vector under threadpool_limits(limits=1).

Section B — Multi-RHS (n_units > 1):
  Models the no-feature-selection primal path in __sub_fit: all output units share
  the same W0 matrix and the RHS is X.T @ Y of shape (n_feat+1, n_units), solved
  in a single call.  Shows whether scipy's Cholesky advantage grows or shrinks
  as batch size increases.

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


def make_normal_equation(
    n_feat: int,
    n_units: int = 1,
    dtype=np.float32,
    n_samples: int = N_SAMPLES,
    alpha: float = ALPHA,
    seed: int = 0,
):
    """Build (W0, rhs) the way __sub_fit (primal path) does.

    W0  : (n_feat+1, n_feat+1) SPD
    rhs : (n_feat+1,) when n_units==1  (feature-selection path)
          (n_feat+1, n_units) otherwise (no-feature-selection primal path)
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_feat)).astype(dtype)
    Xa = np.hstack([X, np.ones((n_samples, 1), dtype=dtype)])  # add bias column
    W0 = Xa.T @ Xa + alpha * np.eye(n_feat + 1, dtype=dtype)
    Y = rng.standard_normal((n_samples, n_units)).astype(dtype)
    rhs = Xa.T @ Y  # (n_feat+1, n_units)
    if n_units == 1:
        rhs = rhs[:, 0]  # (n_feat+1,) — matches feature-selection path exactly
    return W0, rhs


def bench_solve(solver_fn, W0, rhs, n_repeats: int = N_REPEATS, warmup: int = WARMUP):
    """Return median wall-time in seconds.

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


def section_a(limit_threads: bool):
    print("\n" + "=" * 60)
    print("Section A: Single-RHS (n_units=1), n_feat sweep")
    print("Models feature-selection ON path (per-unit solve)")
    print("ratio = scipy / numpy  (>1 means scipy is slower)")
    print("=" * 60)

    ctx = threadpool_limits(limits=1, user_api="blas") if limit_threads else _nullctx()
    with ctx:
        for dtype in DTYPES:
            print(f"\ndtype={np.dtype(dtype).name}")
            hdr = f"  {'n_feat':>7} {'matrix':>9} {'scipy_us':>10} {'numpy_us':>10} {'ratio':>7}"
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for n_feat in NFEAT_SWEEP:
                W0, rhs = make_normal_equation(n_feat, n_units=1, dtype=dtype)
                dim = W0.shape[0]
                ts = bench_solve(_solve_scipy, W0, rhs)
                tn = bench_solve(_solve_numpy, W0, rhs)
                print(
                    f"  {n_feat:>7} {dim:>4}x{dim:<4} "
                    f"{ts * 1e6:>9.1f} {tn * 1e6:>9.1f} {ts / tn:>7.2f}"
                )


def section_b(limit_threads: bool):
    print("\n" + "=" * 60)
    print("Section B: Multi-RHS (batched), n_feat x n_units grid")
    print("Models no-feature-selection primal path (single batched solve)")
    print("ratio = scipy / numpy  (>1 means scipy is slower)")
    print("=" * 60)

    ctx = threadpool_limits(limits=1, user_api="blas") if limit_threads else _nullctx()
    with ctx:
        # B1: ratio grid
        for dtype in DTYPES:
            print(f"\ndtype={np.dtype(dtype).name}  — ratio grid")
            col_hdr = f"  {'n_feat':>7} |" + "".join(f" {u:>8}" for u in N_UNITS_SWEEP)
            units_hdr = "  n_units→  |" + "".join(f" {u:>8}" for u in N_UNITS_SWEEP)
            print(units_hdr)
            print("  " + "-" * (len(col_hdr) - 2))
            for n_feat in NFEAT_SWEEP:
                row = f"  {n_feat:>7} |"
                for n_units in N_UNITS_SWEEP:
                    W0, rhs = make_normal_equation(n_feat, n_units=n_units, dtype=dtype)
                    ts = bench_solve(_solve_scipy, W0, rhs)
                    tn = bench_solve(_solve_numpy, W0, rhs)
                    row += f" {ts / tn:>8.2f}"
                print(row)

        # B2: per-unit time breakdown
        print("\n" + "-" * 60)
        print("Section B2: per-unit time (total_ms / n_units)")
        print("Shows whether scipy amortises better with larger batch")
        print("-" * 60)
        for dtype in DTYPES:
            print(f"\ndtype={np.dtype(dtype).name}")
            hdr = (
                f"  {'n_feat':>7} {'n_units':>8} "
                f"{'scipy_tot_us':>13} {'numpy_tot_us':>13} "
                f"{'scipy/unit_us':>14} {'numpy/unit_us':>14} "
                f"{'ratio':>7}"
            )
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for n_feat in NFEAT_SWEEP:
                for n_units in N_UNITS_SWEEP:
                    W0, rhs = make_normal_equation(n_feat, n_units=n_units, dtype=dtype)
                    ts = bench_solve(_solve_scipy, W0, rhs)
                    tn = bench_solve(_solve_numpy, W0, rhs)
                    print(
                        f"  {n_feat:>7} {n_units:>8} "
                        f"{ts * 1e6:>13.1f} {tn * 1e6:>13.1f} "
                        f"{ts * 1e6 / n_units:>14.2f} {tn * 1e6 / n_units:>14.2f} "
                        f"{ts / tn:>7.2f}"
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

    section_a(limit_threads=limit)
    section_b(limit_threads=limit)


if __name__ == "__main__":
    main()

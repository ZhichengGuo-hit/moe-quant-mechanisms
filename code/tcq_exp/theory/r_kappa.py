#!/usr/bin/env python
"""Codebook mismatch ratio r(kappa) for a clipped symmetric Laplace source.

Reproduces Section "Codebook mismatch: a heavy-tail distortion ratio" of the
paper.  Every 128-weight group is modelled as a symmetric Laplace with scale
b_w clipped to [-a, a]; four-level symmetric quantization of two grid
conventions is compared against the Lloyd-Max codebook of the same size.

    r(kappa) = D_unif / D_LM,        kappa = b_w / a

Grid conventions
----------------
mid_tread  reconstruction points +-a/4, +-3a/4   (Eq. dunif; Voronoi
           boundaries at 0 and +-a/2)  -- the paper's analytical reference
max_tread  reconstruction points +-a/3, +-a     (the grid implied by the TCQ
           scale convention d_b = max|W| / 1.5)

D_LM is obtained by fixed-point (Lloyd-Max) iteration on a uniform grid using
density-weighted quadrature, starting from the mid-tread points.

Reference values quoted in the paper (mid_tread):
    r(0.05) = 48.2    r(0.20) = 2.34    r(0.30) = 1.52
    r in [2, 5]  <=>  kappa in [0.13, 0.23]
    r(kappa = 0.185) = 2.62   (the E21 unweighted caliber)
and for grid convention max_tread:
    r(0.20) = 4.6

Usage
-----
    python r_kappa.py            # table + self-test (exit 1 on assertion)
    python r_kappa.py --csv      # machine-readable output

No dependencies beyond numpy.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

N_GRID = 200_001          # quadrature points on [-a, a]
N_LLOYD = 200             # Lloyd-Max iterations (converges in < 40)
TOL = 1e-12


def laplace_pdf(kappa: float, a: float = 1.0, n: int = N_GRID):
    """Symmetric Laplace on [-a, a], normalised over the clipped support."""
    b = kappa * a
    w = np.linspace(-a, a, n)
    Z = 2.0 * b * (1.0 - np.exp(-a / b))       # integral of exp(-|w|/b)
    return w, np.exp(-np.abs(w) / b) / Z


def distortion(centroids, kappa: float, a: float = 1.0) -> float:
    """E[(W - Q(W))^2] for nearest-neighbour (Voronoi) assignment."""
    w, p = laplace_pdf(kappa, a)
    c = np.asarray(centroids, dtype=float)
    idx = np.abs(w[:, None] - c[None, :]).argmin(axis=1)
    return float(np.trapezoid(p * (w - c[idx]) ** 2, w))


def lloyd_max(kappa: float, init, a: float = 1.0, iters: int = N_LLOYD):
    """Lloyd-Max fixed point: centroids = conditional means of their cells."""
    w, p = laplace_pdf(kappa, a)
    c = np.asarray(init, dtype=float).copy()
    for _ in range(iters):
        idx = np.abs(w[:, None] - c[None, :]).argmin(axis=1)
        new = np.empty_like(c)
        for j in range(c.size):
            m = idx == j
            # an empty cell keeps its previous centroid (cannot happen here)
            new[j] = (np.trapezoid(p[m] * w[m], w[m]) / np.trapezoid(p[m], w[m])
                      if m.any() else c[j])
        if np.allclose(new, c, atol=TOL):
            return new
        c = new
    raise RuntimeError(f"Lloyd-Max did not converge at kappa={kappa}")


def uniform_points(kappa: float, convention: str, a: float = 1.0):
    if convention == "mid_tread":
        return np.array([-3 * a / 4, -a / 4, a / 4, 3 * a / 4])
    if convention == "max_tread":
        return np.array([-a, -a / 3, a / 3, a])
    raise ValueError(convention)


def ratio(kappa: float, convention: str = "mid_tread", a: float = 1.0) -> float:
    """r = D_unif / D_LM at tail parameter kappa."""
    d_unif = distortion(uniform_points(kappa, convention, a), kappa, a)
    c_lm = lloyd_max(kappa, uniform_points(kappa, "mid_tread", a), a)
    return d_unif / distortion(c_lm, kappa, a)


def kappa_for_ratio(target: float, lo: float = 0.02, hi: float = 0.35,
                     tol: float = 1e-6) -> float:
    """Bisection on the (monotone decreasing) r(kappa) curve."""
    f_lo = ratio(lo) - target
    if f_lo < 0:
        raise ValueError(f"no crossing below kappa={hi} for r={target}")
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if ratio(mid) - target > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", action="store_true", help="CSV instead of a table")
    args = ap.parse_args(argv)

    grid = [0.05, 0.10, 0.13, 0.15, 0.185, 0.20, 0.23, 0.25, 0.30]
    rows = []
    for k in grid:
        rows.append((k, ratio(k, "mid_tread"), ratio(k, "max_tread")))

    if args.csv:
        print("kappa,r_mid_tread,r_max_tread")
        for k, r_mid, r_max in rows:
            print(f"{k:.3f},{r_mid:.4f},{r_max:.4f}")
    else:
        print(f"{'kappa':>6}  {'r(mid-tread)':>12}  {'r(max-tread)':>12}")
        for k, r_mid, r_max in rows:
            print(f"{k:6.3f}  {r_mid:12.3f}  {r_max:12.3f}")
        lo = kappa_for_ratio(5.0)
        hi = kappa_for_ratio(2.0)
        print(f"\nr in [2, 5]  <=>  kappa in [{lo:.3f}, {hi:.3f}]")
        print(f"r(kappa=0.185) = {ratio(0.185):.4f}   (E21 unweighted 2.62)")
        print(f"max-tread r(kappa=0.20) = {ratio(0.20, 'max_tread'):.4f}")

    # ---- unit tests: the values quoted in the paper ----------------------
    checks = [
        ("r(0.05) = 48.2", ratio(0.05), 48.2, 0.3),
        ("r(0.20) = 2.34", ratio(0.20), 2.34, 0.02),
        ("r(0.30) = 1.52", ratio(0.30), 1.52, 0.02),
        ("r(0.185) = 2.62", ratio(0.185), 2.62, 0.02),
        ("max-tread r(0.20) = 4.6", ratio(0.20, "max_tread"), 4.6, 0.1),
        ("kappa(r=5) = 0.13", kappa_for_ratio(5.0), 0.13, 0.005),
        ("kappa(r=2) = 0.23", kappa_for_ratio(2.0), 0.23, 0.01),
    ]
    failed = 0
    for label, got, want, tol in checks:
        ok = abs(got - want) <= tol
        failed += not ok
        if not args.csv:
            print(f"[{'ok' if ok else 'FAIL'}] {label:<26} got {got:.4f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Composition- and force-resolved validation errors for the Na-P-S ACE potential.

Reproduces the composition/force-tier RMSE breakdown reported in the Supporting
Information (Table S: "Composition- and Force-Resolved Training Errors").

For every structure in the training dataset the ACE energy and forces are
predicted and compared with the r2SCAN reference labels. Energy errors are
reported after a per-element linear reference alignment (consistent with the
aggregate values). Configurations are grouped by composition class
(Na metal-rich / Na-P binary / Na-S binary / Na-P-S ternary) and by a force
tier (near-equilibrium |F|<3 eV/A vs reactive |F|>=3 eV/A).

Requires: pyace (ML-PACE python interface), ase, numpy, pandas.
Run in the ACE environment, e.g.:

    python validation.py \
        --potential ../mlip/potential.yaml \
        --active-set ../mlip/potential.asi \
        --dataset naps_collected_20k.pckl.gzip \
        --out validation_results.npz
"""
import argparse
import time
import numpy as np
import pandas as pd

ELEMENTS = ["Na", "P", "S"]
FORCE_THRESHOLD = 3.0  # eV/A: reactive / non-equilibrium threshold
NA_METAL_XNA = 0.90    # x_Na above which a cell is "Na metal-rich"
TRACE = 0.03           # atom-fraction below which a species is treated as absent


def classify(x_na, x_p, x_s):
    """Assign a structure to a composition class from its stoichiometry."""
    if x_na > NA_METAL_XNA:
        return "Na metal-rich"
    if x_s < TRACE and x_p > 0:
        return "Na-P (binary)"
    if x_p < TRACE and x_s > 0:
        return "Na-S (binary)"
    if x_p > 0 and x_s > 0:
        return "Na-P-S (ternary)"
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--potential", required=True, help="ACE potential .yaml")
    ap.add_argument("--active-set", required=True, help="ACE active set .asi")
    ap.add_argument("--dataset", required=True,
                    help="pickled dataframe (gzip) with columns "
                         "ase_atoms, energy, forces, max_forces")
    ap.add_argument("--out", default="validation_results.npz")
    args = ap.parse_args()

    from pyace import PyACECalculator
    calc = PyACECalculator(basis_set=args.potential)
    calc.set_active_set(args.active_set)

    d = pd.read_pickle(args.dataset, compression="gzip")
    n = len(d)
    e_dft = np.zeros(n)
    e_pred = np.zeros(n)
    nat = np.zeros(n, int)
    counts = np.zeros((n, 3))
    fmae = np.zeros(n)
    frmse = np.zeros(n)
    x_na = np.zeros(n)
    x_p = np.zeros(n)
    x_s = np.zeros(n)

    t0 = time.time()
    for i, (at, edft, fdft) in enumerate(zip(d["ase_atoms"], d["energy"], d["forces"])):
        at = at.copy()
        at.calc = calc
        e_pred[i] = at.get_potential_energy()
        e_dft[i] = edft
        df = (at.get_forces() - np.array(fdft)).ravel()
        fmae[i] = np.abs(df).mean()
        frmse[i] = np.sqrt((df ** 2).mean())
        sym = at.get_chemical_symbols()
        nat[i] = len(sym)
        for j, e in enumerate(ELEMENTS):
            counts[i, j] = sym.count(e)
        x_na[i], x_p[i], x_s[i] = counts[i] / nat[i]
    print(f"predicted all {n} structures in {time.time() - t0:.0f}s")

    # per-element reference alignment on (E_pred - E_dft)
    resid = e_pred - e_dft
    coef, *_ = np.linalg.lstsq(counts, resid, rcond=None)
    print("per-element offset (eV/atom):", dict(zip(ELEMENTS, coef.round(4))))
    e_err = (resid - counts @ coef) / nat * 1000.0  # signed meV/atom
    abs_e = np.abs(e_err)
    print(f"\nGLOBAL: E MAE={abs_e.mean():.2f} RMSE={np.sqrt((e_err**2).mean()):.2f} meV/atom | "
          f"F MAE={fmae.mean()*1000:.2f} RMSE={np.sqrt((frmse**2).mean())*1000:.2f} meV/A")

    max_f = d["max_forces"].values
    cats = np.array([classify(x_na[i], x_p[i], x_s[i]) for i in range(n)])
    tier = np.where(max_f >= FORCE_THRESHOLD, "reactive (|F|>=3)", "near-equilib (|F|<3)")

    order = ["Na metal-rich", "Na-P (binary)", "Na-S (binary)", "Na-P-S (ternary)", "other"]
    hdr = "{:<18} {:<20} {:>6} {:>9} {:>9} {:>9} {:>9}"
    row_fmt = "{:<18} {:<20} {:>6d} {:>9.2f} {:>9.2f} {:>9.2f} {:>9.2f}"
    print("\n" + hdr.format("composition", "force tier", "n",
                            "E_MAE", "E_RMSE", "F_MAE", "F_RMSE"))
    for c in order:
        for tlabel in ["near-equilib (|F|<3)", "reactive (|F|>=3)"]:
            m = (cats == c) & (tier == tlabel)
            if m.sum() == 0:
                continue
            print(row_fmt.format(c, tlabel, int(m.sum()), abs_e[m].mean(),
                                 np.sqrt((e_err[m] ** 2).mean()),
                                 fmae[m].mean() * 1000, np.sqrt((frmse[m] ** 2).mean()) * 1000))
        m = (cats == c)
        if m.sum():
            print(row_fmt.format(c, "ALL", int(m.sum()), abs_e[m].mean(),
                                 np.sqrt((e_err[m] ** 2).mean()),
                                 fmae[m].mean() * 1000, np.sqrt((frmse[m] ** 2).mean()) * 1000))
            print()

    np.savez(args.out, e_err=e_err, fmae=fmae, frmse=frmse, cats=cats, tier=tier,
             max_f=max_f, x_na=x_na, x_p=x_p, x_s=x_s, nat=nat)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

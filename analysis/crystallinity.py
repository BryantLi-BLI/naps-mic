#!/usr/bin/env python3
"""
Crystallinity quantification of the interphase Na2S region.

Compares the Na-S partial radial distribution function (RDF) of the Na2S-rich
interphase slab (extracted from the interface MD trajectory, averaged over the
final frames) against a crystalline antifluorite Na2S reference. Reproduces the
metrics reported in the Supporting Information (Fig. S "Crystallinity of the
Interphase Na2S Region"):

- first Na-S coordination shell position,
- relative height of the second/third-neighbor peaks, and
- a translational-order parameter tau = <|g(r) - 1|> over 3-10 A.

The interphase Na2S shows a sharp first shell but ~5-fold suppressed long-range
order relative to the crystal, i.e. it is amorphous rather than nanocrystalline.

Requires: pymatgen, pymatgen-analysis-diffusion, numpy.

    python crystallinity.py \
        --dump last_frames.dump \
        --reference reference/Na2S.cif \
        --zmin 460 --zmax 490 \
        --out crystallinity_rdf.npz
"""
import argparse
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.diffusion.aimd.rdf import RadialDistributionFunctionFast

RMAX, NGRID, SIGMA = 10.0, 501, 0.1
TAU_LO, TAU_HI = 3.0, 10.0  # translational-order integration window (A)


def parse_frames(path):
    """Parse a LAMMPS text dump (ITEM: TIMESTEP ... format) into frames."""
    frames = []
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        assert lines[i].startswith("ITEM: TIMESTEP"), lines[i]
        nat = int(lines[i + 3])
        box = [list(map(float, lines[i + 5 + k].split())) for k in range(3)]
        start = i + 9
        frames.append((box, lines[start:start + nat]))
        i = start + nat
    return frames


def slab_structure(box, atoms, zmin, zmax):
    """Build a pymatgen Structure from the Na/S atoms in the z-slab [zmin, zmax)."""
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = box
    lx, ly = xhi - xlo, yhi - ylo
    els, xyz = [], []
    for ln in atoms:
        p = ln.split()
        el = p[1]
        x, y, z = float(p[2]), float(p[3]), float(p[4])
        if zmin <= z < zmax and el in ("Na", "S"):
            els.append(el)
            xyz.append((x - xlo, y - ylo, z - zmin))
    xyz = np.array(xyz)
    c = zmax - zmin
    lat = Lattice.from_parameters(lx, ly, c, 90, 90, 90)
    return Structure(lat, els, xyz / np.array([lx, ly, c]), coords_are_cartesian=False)


def partial_rdf(structs, a, b):
    """Average Na-S partial RDF over a list of structures."""
    gs, r = [], None
    for s in structs:
        rf = RadialDistributionFunctionFast(structures=[s], ngrid=NGRID, rmax=RMAX, sigma=SIGMA)
        r, g = rf.get_rdf(a, b)
        gs.append(np.array(g))
    return np.array(r), np.mean(gs, axis=0)


def analyze(r, g, label):
    """Report first-shell position, normalized tail peak height, and tau."""
    i1 = np.argmax(g[r < 4.0])
    r1, g1 = r[i1], g[i1]
    gn = g / g1
    tail_max = gn[r > 4.0].max()
    mo = (r >= TAU_LO) & (r <= TAU_HI)
    tau = np.trapz(np.abs(g[mo] - 1.0), r[mo]) / (TAU_HI - TAU_LO)
    print(f"[{label}] 1st-shell r={r1:.2f} A g={g1:.2f} | "
          f"max(g_norm, r>4A)={tail_max:.3f} | tau(3-10A)={tau:.3f}")
    return gn


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True,
                    help="LAMMPS text dump of the interphase frames to average")
    ap.add_argument("--reference", default="reference/Na2S.cif",
                    help="crystalline Na2S reference structure")
    ap.add_argument("--zmin", type=float, default=460.0, help="Na2S-rich slab lower z (A)")
    ap.add_argument("--zmax", type=float, default=490.0, help="Na2S-rich slab upper z (A)")
    ap.add_argument("--supercell", type=int, default=4,
                    help="reference supercell replication so RMAX fits")
    ap.add_argument("--out", default="crystallinity_rdf.npz")
    args = ap.parse_args()

    # interphase slab (amorphous candidate)
    frames = parse_frames(args.dump)
    slabs = [slab_structure(b, a, args.zmin, args.zmax) for b, a in frames]
    comp = slabs[-1].composition.as_dict()
    nna, ns = comp.get("Na", 0), comp.get("S", 0)
    print(f"slab atoms (last frame): Na={int(nna)} S={int(ns)} Na:S={nna/max(ns,1):.2f}")
    r, g_int = partial_rdf(slabs, "Na", "S")

    # crystalline reference
    xtal = Structure.from_file(args.reference)
    xtal.make_supercell([args.supercell] * 3)
    _, g_xtal = partial_rdf([xtal], "Na", "S")

    gn_int = analyze(r, g_int, "interphase Na2S")
    gn_xtal = analyze(r, g_xtal, "crystalline Na2S")

    np.savez(args.out, r=r, g_int=g_int, g_xtal=g_xtal, gn_int=gn_int, gn_xtal=gn_xtal)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

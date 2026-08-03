# Analysis

Scripts that produce the quantitative results and figures. Run in the `ace`
conda environment (`conda env create -f ../environment.yml`); `validation.py`
additionally requires `pyace` installed into that same environment.

| Script | Produces | Key inputs |
| --- | --- | --- |
| `percolation.py` | Na-P bond-graph percolation per frame; cutoff-sensitivity sweeps (Fig. 4, Fig. S) | interface MD dump; cutoffs Na-P 3.2 Å, P-P 2.5 Å |
| `interphase_analysis.py` + `interphase_plotting.py` | z-resolved composition / coordination profile; interphase definition | interface MD trajectory |
| `crystallinity.py` | Na-S partial RDF + translational-order τ of the Na2S region (Fig. S) | interphase frames dump; `reference/Na2S.cif` |
| `validation.py` | composition- and force-resolved potential errors (Table S) | `../mlip/potential.yaml`, `../mlip/potential.asi`, training dataset |
| `onsager/` | Green-Kubo Onsager coefficients and ionic fluxes (Fig. 3, Fig. S) | amorphous melt-quench MD trajectories |

## Notes

- `percolation.py` and `interphase_analysis.py` carry trajectory paths in a
  configuration block near the top of the file; edit these to point at your
  local MD dumps before running.
- `crystallinity.py` and `validation.py` take command-line arguments; run with
  `-h` for usage.
- The MD trajectories these scripts consume are distributed via MPContribs (see
  the top-level `README.md`).

## Onsager submodule

`onsager/run_batch.py` drives the per-composition Green-Kubo analysis
(`onsager_io.py`, `onsager_processing.py`, `onsager_utils.py`) and
`onsager_plotting.py` renders the flux/coefficient figures.

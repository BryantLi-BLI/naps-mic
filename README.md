# Na₃PS₄/Na Mixed Conduction Interphase — MLIP-MD analysis

Machine-learning interatomic potential (ACE), analysis code, and model files
supporting:

> B. Y. Li, H. Jeon, K. A. Persson,
> *Distinguishing the Mixed Conduction Interphase: A Machine Learning Molecular
> Dynamics Study on the Na₃PS₄/Na Battery Interface.*
> Machine Learning: Science and Technology (in review).
> [DOI to be added upon publication]

This repository contains the fitted ACE potential, its training configuration,
and the analysis scripts used to produce the figures and quantitative results.
The large data artifacts (training dataset, DFT reference calculations, and MD
trajectories) are distributed separately (see **Data** below).

## Repository layout

```
mlip/       ACE potential and training configuration
  input.yaml        pacemaker training input (Na-P-S, 350 functions/element, 7.0 A cutoff)
  potential.yaml    final fitted ACE potential
  potential.asi     active set (D-optimality) for extrapolation-grade monitoring
analysis/   analysis used for the paper figures
  percolation.py            Na-P bond-graph percolation + cutoff-sensitivity sweeps
  interphase_analysis.py    z-resolved composition / coordination profile
  interphase_plotting.py    plotting for the interphase analysis
  crystallinity.py          Na-S partial RDF + translational-order (tau) of the Na2S region
  validation.py             composition- and force-resolved potential errors
  onsager/                  Green-Kubo Onsager transport coefficients + ionic fluxes
  reference/Na2S.cif        crystalline antifluorite Na2S reference
md/         production molecular dynamics (see md/README.md)
```

## Data (distributed separately)

These exceed a normal git repository and are hosted externally:

| Artifact | Location |
| --- | --- |
| Training dataset (20,114 structures, r²SCAN labels) | [MPContribs — naps\_mci](https://contribs.materialsproject.org/projects/naps_mci) (public upon publication) |
| Raw DFT reference calculations | MPContribs — naps\_mci (public upon publication) |
| Interface MD trajectories | MPContribs — naps\_mci (public upon publication) |

Download the training dataset next to `mlip/input.yaml` (or update the
`data.filename` field) to retrain, and provide the MD dumps to the analysis
scripts as described in `analysis/README.md`.

## Environment

All scripts run in a single conda environment named `ace`:

```bash
conda env create -f environment.yml   # creates the `ace` env (pymatgen, ase, numpy, pandas, networkx)
conda activate ace
```

The ACE tools (`pyace`, `pacemaker`) required by `mlip/` (retraining) and
`analysis/validation.py` are installed into this same environment separately,
following the [python-ace / pacemaker instructions](https://pacemaker.readthedocs.io).

## Reproducing

- **Retrain the potential:** `pacemaker mlip/input.yaml` (requires the training dataset).
- **Potential accuracy (Table S):** `python analysis/validation.py --potential mlip/potential.yaml --active-set mlip/potential.asi --dataset naps_collected_20k.pckl.gzip`
- **Percolation (Fig. 4, Fig. S):** `python analysis/percolation.py` (edit the trajectory path at the top).
- **Na₂S crystallinity (Fig. S):** `python analysis/crystallinity.py --dump last_frames.dump --reference analysis/reference/Na2S.cif`
- **Onsager transport (Fig. 3, Fig. S):** see `analysis/onsager/run_batch.py`.

## License

Released under the MIT License (see `LICENSE`).

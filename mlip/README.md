# ACE potential and training

- **`input.yaml`** — pacemaker training configuration for the Na-P-S ACE
  potential. Key settings: elements `['Na','P','S']`, 350 basis functions per
  element, 7.0 Å cutoff, Finnis-Sinclair embedding (`ndensity: 8`), SBessel
  radial basis, ZBL repulsion (`repulsion: auto`), 90:10 train/test split.
- **`potential.yaml`** — final fitted ACE potential (use this for MD and for
  single-point predictions).
- **`potential.asi`** — active set (D-optimality), used to compute the
  extrapolation grade (γ) for active-learning / stability monitoring.

## Training dataset

`input.yaml` reads `naps_collected_20k.pckl.gzip` — the 20,114-structure
training set with r²SCAN energy/force labels. This file is distributed via
MPContribs (see the top-level `README.md`); place it in this directory,
or edit the `data.filename` field to point at its location.

## Retraining

```bash
pacemaker input.yaml
```

Requires `pacemaker` (python-ace); see https://pacemaker.readthedocs.io.

# Production molecular dynamics

The production interface simulations were run with **LAMMPS** using the
**ML-PACE (Kokkos) pair style** with the fitted potential in
`../mlip/potential.yaml`.

Verified production settings (see the Supporting Information of the paper):

| Setting | Value |
| --- | --- |
| Interface builder | `CoherentInterfaceBuilder` (pymatgen), Zur–McGill strain minimization |
| System size | 500,000 atoms |
| Ensemble | *NpT* (1 bar), Nosé–Hoover thermostat + barostat (τ = 1 ps) |
| Temperature / timestep | 300 K / 1 fs |
| Duration | 10.5 ns |

The interface-construction workflow and the HPC job scripts are not included
here; the settings above fully specify the production run given the released
potential. The resulting trajectories are distributed via MPContribs (see the
top-level `README.md`) and are the inputs to the scripts in `../analysis/`.

# Figure 4 field export

Figure 4 is intentionally not rendered by this repository. Generate the three
VTK datasets at `u_x = 0.75 mm` by invoking the Figure 3 runner separately for
the local, monolithic-gradient, and staggered-gradient formulations with:

```text
--maximum-displacement 0.75 --export-final
```

Each run writes `kappa`, `d`, and von Mises stress as cell fields. Gradient
runs also write the nodal `phi` field. The adjacent JSON manifest records the
actual displacement, mesh, formulation, and VTK path for ParaView.

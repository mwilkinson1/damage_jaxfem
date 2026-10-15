# JAX-FEM Gradient Damage

## Project overview

This repository implements finite-strain continuum damage models in JAX-FEM.
It accompanies the work by Wilkinson, Amiri-Hezaveh, and Tepole on
gradient-enhanced damage with differentiable, physics-augmented constitutive
models. The packaged closed-form implementation uses a compressible
Neo-Hookean elastic energy and exponential stiffness degradation.

## Gradient regularization

Local strain-softening models can concentrate damage into a mesh-dependent
band. The gradient-enhanced formulation introduces a continuous nonlocal
history field and adds gradient and penalty energies. The gradient term
provides a material length scale, while the penalty term couples the nonlocal
field to the local quadrature-point history variable. Together they spread
damage over a finite region and reduce mesh sensitivity.

## Implementations

- **Local:** displacement equilibrium with a quadrature-point damage history;
  no nonlocal field or gradient regularization.
- **Staggered gradient:** alternates the global displacement/nonlocal-field
  solve with local history updates until both states converge.
- **Monolithic gradient:** the primary formulation. It solves displacement and
  the nonlocal field together while condensing the local history update into
  each global Newton evaluation.

## Citation

```bibtex
@article{wilkinson_differentiable_2026,
  title   = {A differentiable framework for gradient enhanced damage with physics-augmented neural networks in {JAX}-{FEM}},
  author  = {Wilkinson, Mark and Amiri-Hezaveh, Amirhossein and Tepole, Adrian Buganza},
  journal = {Mechanics Research Communications},
  volume  = {156},
  pages   = {104768},
  year    = {2026},
  doi     = {10.1016/j.mechrescom.2026.104768}
}
```

## License

No software license has been specified for this repository.

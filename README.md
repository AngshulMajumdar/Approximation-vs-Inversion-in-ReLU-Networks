# Spline-ReLU inverse-problem reproducibility code

This is a dependency-free Rust reproduction of the numerical experiment in
**Approximation versus Invertibility in Spline-ReLU Models**.

It implements:

- the 256-point periodic Gaussian forward operator;
- fixed-knot one-hidden-layer ReLU spline spaces;
- modified Gram-Schmidt orthonormalization;
- a one-sided Jacobi SVD for restricted singular values and least-squares solves;
- 100 fixed-seed Gaussian-noise trials for each model width;
- CSV output for the numerical table; and
- an SVG version of the three-panel numerical figure.

No external data are used. No BLAS/LAPACK or plotting library is required.

## Run

```text
cargo run --release
```

The program writes:

```text
experiment_results.csv
stability_tradeoff.svg
```

The random generator is deterministic (SplitMix64 plus Box-Muller) with seed
`20260916`. Consequently, repeated Rust runs are bitwise deterministic on the
same floating-point platform. The Monte-Carlo noise stream is independent of
NumPy's original generator, while the forward model, spline spaces, norms,
noise level, widths, trial count, and least-squares problem are the same.

The repository intentionally contains no hidden files, no `.github` directory,
and no `.gitignore` file.

# Workflows: commands, Kaimon, CI, releases

_How to run, test, document and release._ (Layer 3 reference.)

## Kaimon session

- `start_session(project_path = "C:/Users/Nicola/.julia/dev/PatternVectors")`; `kaimon.toml` pins Julia 1.13.
- First call: `ex(q = false, e = "(VERSION, Base.active_project(), pathof(PatternVectors))")`.
- The AD backends are test dependencies: AD checks go through `run_tests()`.
- Do not run `julia --project` scripts that precompile this package while the session is open (Revise keeps the old
  code); restart the session afterwards.

## Commands

```bash
julia --project -e 'import Pkg; Pkg.test()'             # test/runtests.jl, then the AD backends
julia --project=benchmark benchmark/runbenchmarks.jl
julia --project=test/reactant -e 'import Pkg; Pkg.instantiate(); include("test/reactant/runtests.jl")'  # Reactant (XLA), local only; REACTANT_BACKEND=cuda on Linux
julia --project=docs docs/make.jl                      # Documenter site, into docs/build
julia -e 'using JuliaFormatter; format(".")'            # uses .JuliaFormatter.toml
```

## CI

`CI.yml`: Julia 1.11 and 1.12, ubuntu, with coverage (Codecov). `Benchmark.yml`: the suite on 1.12, results as an
artifact and job summary. `DeployPage.yml`: the docs through the Pages artifact (no `gh-pages` branch, #14).
`CompatHelper.yml`. The default branch is `main`.

## Sub-environments

`test/`, `test/reactant/`, `docs/` and `benchmark/` load the package through `[sources]` with a path, and each has a `[compat]` entry for
the package itself, so CompatHelper does not open PRs to add one (it has no option to skip a dependency). Bump these
entries in the same commit as a breaking version bump, otherwise the local package no longer satisfies them and the
environment fails to resolve.

## Release

Bump `version` in `Project.toml`, push to `main`, wait for CI, then register the commit on the private registry
`localregistry` (`rcalxrc08/LocalRegistry`, not General) with `LocalRegistry.register`, from Julia 1.12, where
LocalRegistry.jl is installed (`julia +1.12`; registry commits are named `New version: PatternVectors vX.Y.Z`, pushed to
the registry's GitHub remote); then tag `vX.Y.Z` on the registered commit and create the GitHub release (the private
registry has no TagBot). Of the author's packages only FinancialMonteCarlo and FinancialToolbox are in General;
FinancialFFT and VibratoMonteCarlo are in `localregistry` too, FinancialCalibrator is not registered.

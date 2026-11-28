# Tests and CI

How the suite is organized, how to run a part of it, what CI runs.

## Inputs

| Load | Why |
|:--|:--|
| `runtests.jl` | the patterns, the broadcasts ("Nested mixed broadcasts" pins the memory), `dot` (every pattern with ranges, views, integer and complex arrays; no array of the products); then includes the AD files |
| `test_Zygote.jl` | Zygote per pattern, the ChainRules pullbacks called directly, keyword `sum`, deferred cotangents, `dot` |
| `test_Mooncake.jl` | structural tangent, the mixed-broadcast rule (also with a non-default style, Int8 weights, `ZeroPattern`), allocations, the `dot` rule |
| `test_ReverseDiff.jl` | broadcast style, gradients, `dot` |
| `test_Enzyme.jl` | Enzyme vs Zygote (scalars), vs Mooncake (pattern gradients), size independence, `dot` (reverse and forward) |
| `cuda_tests.jl` | Zygote and Mooncake on `CuArray`s, `dot` (every pattern, views, gradients); needs a GPU, not run by `runtests.jl` |
| `reactant/runtests.jl` | Reactant (XLA): values against plain Julia, no constant of `n` values in the program, `dot` with constant and integer weights, traced pattern values, Enzyme; own environment, not run by `runtests.jl` nor on CI |
| `context.jl` | the check that keeps the context layers in sync with the code |
| `references/testing.md` | what each file checks, reference values, local versus CI differences |
| `references/workflows.md` | commands, CI matrix, sub-environments |

## Process

- A new pattern gets a testset in `runtests.jl` and one in each AD file.
- Compare allocations with `@allocated` on a function called once before, inside a function: at the top level the
  globals inflate them.
- Fast loop: `include` the one test file in the project's Kaimon session; before a commit: `run_tests()`; CI on 1.11
  and 1.12.
- `test_z*.jl`, `runbench.jl`, `bench_zygote_mooncake.jl` are untracked local scratch files, not part of the suite.

## Outputs

A green `Pkg.test()`; `context.jl` green whenever a file is added, moved or renamed.

## Verify (Kaimon)

| Check | Expect |
|:--|:--|
| `run_tests()` summary | pass; reports the Julia of `kaimon.toml`. Locally on 1.13, `test_Enzyme.jl` may abort (`references/testing.md`) |
| `include("test/context.jl")` | pass |

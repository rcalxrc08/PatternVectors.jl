# etc: README benchmarks and logo

## Inputs

| Load | Why |
|:--|:--|
| `benchmarks.jl` | the README / `docs/src/performance.md` table: dense weights vs `PatternVector`, forward and gradients (Zygote, Mooncake, Enzyme) |
| `references/performance.md` | load conditions of the last run, how to compare |

## Process

- Run in an environment with PatternVectors, BenchmarkTools, Zygote, Mooncake and Enzyme; the table in
  `docs/src/performance.md` and the chart data in `docs/figures.jl` are updated together.
- `logo.png` and `favicon.ico` are the only copies; `docs/make.jl` copies them into the docs assets. `logo2.svg` is
  untracked.

## Outputs

The performance table of the README and of the docs.

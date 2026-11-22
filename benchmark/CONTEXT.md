# Benchmark suite

The BenchmarkTools suite run by CI (`Benchmark.yml`, Julia 1.12) and by hand.

## Inputs

| Load | Why |
|:--|:--|
| `benchmarks.jl` | the suite: every pattern, several element types (Float16, Float64, irrationals, duals), functions and gradients |
| `runbenchmarks.jl` | runs the suite, tunes into `tune.json`, writes `results.json` / `results.md` (git-ignored) and the job summary |
| `references/performance.md` | the last measured numbers and how to measure |

## Process

- CI numbers are for trends only; numbers written down come from an idle machine, one fresh process per variant, or
  interleaved in one process (`references/performance.md`).
- A change to a hot path is timed before and after (the old commit in a `git worktree`).

## Outputs

`results.md`, uploaded as a CI artifact; an updated table in `references/performance.md` or `docs/src/performance.md`
when a number changes.

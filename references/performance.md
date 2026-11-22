# Performance

_Where the measured numbers live and how to measure._ (Layer 3 reference; `benchmark/` and `etc/benchmarks.jl` update
the numbers.)

| Numbers | Where | Conditions |
|:--|:--|:--|
| Dense vs PatternVector, forward and gradients (Zygote, Mooncake, Enzyme), n = 10^5 | `docs/src/performance.md`, README "Performance", chart data in `docs/figures.jl` | Julia 1.13, Intel i7-8550U, single thread; regenerated 2026-11-02 at about 16% load (every column ~40% slower than the previous table: compare within the table) |
| Cotangent of `sum`: package rule vs Zygote's, vs FillArrays | `docs/src/ad.md` | n = 10^6, Zygote 0.7.18 |
| Mooncake rule versions, fused vs eager, CPU and GPU | `mooncake.md` | 2026-10-29 to 2026-11-01, MX150 for the GPU |
| `dot_rrule` | `autodiff.md` | n = 10^6 |
| CI suite | `Benchmark.yml` artifact | trends only |

How to measure:

- Timings vary by up to 2x between processes while other Julia processes run: compare variants in one process,
  interleaved, or in fresh processes on an idle machine.
- Hand-written `f` hide boxing: real expressions go through `Broadcast.flatten` closures, so compare with those.
- On `sum(sin.(w .* x) .* exp.(w))` Mooncake is behind Zygote and Enzyme: the fused expression is recomputed per
  argument (`mooncake.md`, "Trade-off").

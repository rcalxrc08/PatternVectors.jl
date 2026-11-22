# Stage 05: linear algebra

`dot` with broadcasts and a reduction only, so that it runs on the device of a dense argument, with its own rules.

## Inputs

| Load | Why |
|:--|:--|
| `linearalgebra.jl` | `dot` for (pattern, dense), (dense, pattern), (pattern, pattern), `check_dot_lengths`, `dot_rrule` |
| `references/autodiff.md` | how ChainRules reaches `dot` (#4) and the measurements of `dot_rrule` (#21) |

## Process

- No element-by-element indexing: the generic `dot` falls back on scalar indexing of GPU arrays (#4).
- `dot_rrule`: no `w .* y` in the forward pass, the cotangent of the dense argument a `PatternVector`, none for integer
  or Bool weights.
- New `dot` methods must not add ambiguities among the package's methods (tested); ReverseDiff's pairs live in
  `ext/ReverseDiffExt.jl`.

## Outputs

`dot` equal to the dense `dot`, real and complex. Checked by `test/runtests.jl` ("dot"), `test/test_Zygote.jl`,
`test/cuda_tests.jl`.

## Verify (Kaimon)

Setup: `using PatternVectors, LinearAlgebra`; `p = PatternVector(7, FillPattern(2.5))`, `y = randn(7)`.

| Check | Expect |
|:--|:--|
| `dot(p, y) ≈ dot(collect(p), y)` | `true` |
| `dot(p, randn(8))` | `DimensionMismatch` |

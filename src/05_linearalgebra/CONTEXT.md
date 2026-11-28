# Stage 05: linear algebra

`dot` with broadcasts and a reduction only, so that it runs on the device of a dense argument, with its own rules.

## Inputs

| Load | Why |
|:--|:--|
| `linearalgebra.jl` | `dot` for (pattern, dense), (dense, pattern), (pattern, pattern), `check_dot_lengths`, `pattern_dot`, `reduce_dot`, `dot_rrule` |
| `references/autodiff.md` | how ChainRules reaches `dot` (#4) and the measurements of `dot_rrule` (#21) |

## Process

- No element-by-element indexing: the generic `dot` falls back on scalar indexing of GPU arrays (#4).
- With a dense array, `reduce_dot` reduces the lazy broadcast in the unwrapped style of the dense array: no array of
  the products, GPUArrays' `mapreduce` on a GPU (D-009). Its Mooncake rule is in `ext/MooncakeExt.jl`.
- `dot_rrule`: no `w .* y` in the forward pass, the cotangent of the dense argument a `PatternVector`, none for integer
  or Bool weights.
- New `dot` methods must not add ambiguities among the package's methods (tested); ReverseDiff's pairs live in
  `ext/ReverseDiffExt.jl`.

## Outputs

`dot` equal to the dense `dot`, real and complex, for any dense argument (ranges, views, integers; sparse is ambiguous, #25), without an
array of the products. Checked by `test/runtests.jl` ("dot"), `test/test_Zygote.jl`, `test/test_Mooncake.jl`
("Mooncake rule for dot"), `test/test_Enzyme.jl`, `test/cuda_tests.jl`.

## Verify (Kaimon)

Setup: `using PatternVectors, LinearAlgebra`; `p = PatternVector(7, FillPattern(2.5))`, `y = randn(7)`.

| Check | Expect |
|:--|:--|
| `dot(p, y) ≈ dot(collect(p), y)` | `true` |
| `dot(p, randn(8))` | `DimensionMismatch` |
| `q = PatternVector(10^5 + 1, EvenOddPattern(1.0, 2.0)); z = randn(10^5 + 1); dot(q, z); @allocated dot(q, z)` | below 1000 |

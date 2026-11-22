# Stage 04: patterns

One file per pattern, each implementing the pattern interface.

## Inputs

| Load | Why |
|:--|:--|
| `zeropattern.jl` | `ZeroPattern`: all zeros, no stored value |
| `fillpattern.jl` | `FillPattern(a)`: `[a, a, …]` |
| `evenoddpattern.jl` | `EvenOddPattern(a, b)`: `[a, b, a, b, …]` |
| `initialvaluepattern.jl` | `InitialValuePattern`: a first value, then a constant |
| `finalvaluepattern.jl` | `FinalValuePattern`: a constant, then a last value |
| `paddedfillpattern.jl` | `PaddedFillPattern`: first and last values around a constant (trapezoid weights) |
| `paddedevenoddpattern.jl` | `PaddedEvenOddPattern(a, b, c, d)`: `[a, b, c, b, c, …, d]` (Simpson weights) |
| `references/public_api.md` | the exported patterns and the interface |
| `references/autodiff.md` | the constructor rules and `pattern_to_vector_pullback` |

## Process

- A pattern implements `pattern_minimum_size`, `getindex_pattern`, `getindex_pattern_range`, `materialize_pattern`,
  `Base.sum`, a ChainRulesCore `rrule` for its constructor and `pattern_to_vector_pullback` ("Defining a new pattern"
  in the README; a pattern can live outside the package).
- All fields have the element type `T`: Mooncake and Enzyme differentiate the fields structurally, and the broadcast
  rebuilds patterns with the positional constructor.
- A new pattern gets its testset in `test/runtests.jl`, a line in `references/public_api.md`, the export and the docs.
- `symmetricpaddedfillpattern.jl` in this folder is an untracked experiment, not included by the module.

## Outputs

The exported patterns. Checked by `test/runtests.jl`, `test/test_Zygote.jl`, `test/test_Mooncake.jl`.

## Verify (Kaimon)

Setup: `using PatternVectors`.

| Check | Expect |
|:--|:--|
| `collect(PatternVector(6, PaddedEvenOddPattern(1, 4, 2, 1)))` | `[1, 4, 2, 4, 2, 1]` |
| `sum(PatternVector(10^18, FillPattern(1)))` | `10^18`, O(1) |

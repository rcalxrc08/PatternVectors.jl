# Public API

_The exported names and the pattern interface._ (Layer 3 reference; `docs/` checks it with `checkdocs = :exports`.)

| Name | Signature | Stage | Notes |
|:--|:--|:--|:--|
| `PatternVector` | `PatternVector(n::Int64, pattern::AbstractPattern{T})` | 01 | `AbstractVector{T}`; `DomainError` below the pattern's minimum length |
| `ZeroPattern` | `ZeroPattern(x)` | 04 | all zeros of the type of `x`, no stored value |
| `FillPattern` | `FillPattern(a)` | 04 | `[a, a, …]` |
| `EvenOddPattern` | `EvenOddPattern(a, b)` | 04 | `[a, b, a, b, …]`, minimum length 2 |
| `InitialValuePattern` | `InitialValuePattern(a, b)` | 04 | a first value, then a constant |
| `FinalValuePattern` | `FinalValuePattern(a, b)` | 04 | a constant, then a last value |
| `PaddedFillPattern` | `PaddedFillPattern(a, b, c)` | 04 | first and last values around a constant (trapezoid weights) |
| `PaddedEvenOddPattern` | `PaddedEvenOddPattern(a, b, c, d)` | 04 | `[a, b, c, b, c, …, d]` (Simpson weights) |

Extended Base and LinearAlgebra functions: `getindex` (ranges return a `PatternVector`), `sum` (O(1)), broadcasting
(stage 02), `LinearAlgebra.dot` (stage 05).

Pattern interface (not exported, documented in the README "Defining a new pattern"): `pattern_minimum_size`,
`getindex_pattern`, `getindex_pattern_range`, `materialize_pattern`, `determine_mixed_pattern_defer` (or
`UnknownPattern`), `Base.sum`, a constructor `rrule` and `pattern_to_vector_pullback`.

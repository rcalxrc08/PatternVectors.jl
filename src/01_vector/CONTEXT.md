# Stage 01: the vector

The `PatternVector{T, P}` type and the interface every pattern implements.

## Inputs

| Load | Why |
|:--|:--|
| `patternvector.jl` | `AbstractPattern{T}`, `PatternVector(n, pattern)` (length check, `@noinline throw_length_error`), `size`, `getindex` (index, `:`, ranges), the `Core.kwcall` forwarding of `sum(x; dims = :)` |
| `references/conventions.md` | before adding a method |
| `references/autodiff.md` | why the length error is out of line (Enzyme#4053) |

## Process

- A `PatternVector` stores `n::Int64` and the pattern only; it must stay `isbits` for `isbits` pattern values (GPU
  kernels capture it by value).
- `getindex` with a range returns a `PatternVector` (through the pattern's `getindex_pattern_range`); `sum` is O(1).
- The keyword `sum(x; dims = :)` (called by Zygote's `unbroadcast`) is forwarded to the positional, pattern-specialized
  `sum`; without it the keyword form is O(n).
- Error paths stay out of line: an inlined throw puts allocations into every broadcast `materialize` (Enzyme#4053).

## Outputs

`PatternVector` with the `AbstractArray` interface. Checked by `test/runtests.jl` (one testset per pattern).

## Verify (Kaimon)

Setup: `using PatternVectors, Test`; `p = PatternVector(6, EvenOddPattern(1.0, 2.0))`.

| Check | Expect |
|:--|:--|
| `isbits(p)`, `sum(p)` | `true`, `9.0` |
| `@inferred p[2:5]` | a `PatternVector{Float64, EvenOddPattern{Float64}}` |
| `PatternVector(1, EvenOddPattern(0, 0))` | `DomainError` |

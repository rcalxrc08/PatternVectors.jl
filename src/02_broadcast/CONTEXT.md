# Stage 02: broadcast

The two broadcast styles: pattern-only broadcasts evaluated on the pattern values, mixed broadcasts fused into one loop
on the dense array's device.

## Inputs

| Load | Why |
|:--|:--|
| `broadcast.jl` | `ArrayStylePatternVector{P}`, `PatternMixtureArrayStyle{N, T}`, `materialize_pattern` calls, `materialize_if_needed`, `materialize_mixture`, `get_style`, `similar` |
| `references/broadcast.md` | the promises of `materialize_if_needed` and what breaks them |
| `references/mooncake.md` | `materialize_mixture` is the hook of the Mooncake rule |

## Process

- Keep the promises of `references/broadcast.md`; the "Nested mixed broadcasts" tests pin them, memory included.
- Pattern-only broadcasts of different patterns combine through `determine_mixed_pattern_defer`; nested
  `PatternMixtureArrayStyle`s (with `CuArrayStyle`) are unwrapped by `get_style`, so `materialize_mixture` always gets
  the innermost style.
- `materialize_mixture(style, f, args...)` stays the single hook for AD rules: any new path goes through it.
- In-place broadcasts (`d .= …`) do not go through this path.

## Outputs

`materialize` returns a `PatternVector` for pattern-only broadcasts and an array of the dense argument's type for mixed
ones. Checked by `test/runtests.jl` ("Nested mixed broadcasts" and the per-pattern broadcast tests), `test/cuda_tests.jl`.

## Verify (Kaimon)

Setup: `using PatternVectors, Test`; `p = PatternVector(10, EvenOddPattern(1.0, 2.0))`, `x = rand(10)`.

| Check | Expect |
|:--|:--|
| `typeof(sin.(p) .* p .+ 1)` | `PatternVector{Float64, EvenOddPattern{Float64}}` |
| `typeof(p .* x .+ 1)` | `Vector{Float64}` |
| `run_tests()` | "Nested mixed broadcasts" passes, memory bounds included |

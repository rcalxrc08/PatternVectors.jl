# Extensions

One extension per weak dependency; the package loads and works without any of them.

## Inputs

| Load | Why |
|:--|:--|
| `MooncakeExt.jl` | the reverse rule of `materialize_mixture` (mixed broadcasts): `arg_role`, one fused NDual pass per differentiable argument, `Partial` |
| `ReverseDiffExt.jl` | `TrackedStyle` wins over both PatternVector styles; `dot` with a `TrackedArray` recorded as ReverseDiff does |
| `SparseArraysExt.jl` | a PatternVector with a sparse vector promotes to the sparse style |
| `references/mooncake.md` | the design of the Mooncake rule, the alternatives evaluated, the cost model and the measurements |
| `references/autodiff.md` | which backend needs what, upstream issues |
| `references/dependencies.md` | why each dependency is weak, compat floors |

## Process

- Mooncake: no rule for `PatternVector` itself (its derived tangent is structural); never `@from_rrule` the
  ChainRulesCore rules. `Partial` stays `@inline` with `Vararg{Any, M}` (a non-inlined vararg call boxes the elements:
  27 args took 18 s and 6 GB). `arg_role` decides both `_is_primitive` and the pullback; dispatching on the primal type
  broke `ZeroPattern` PatternVectors.
- ReverseDiff ≥ 1.18: `TrackedStyle` is an `AbstractArrayStyle`, so both argument orders must defer to it.
- Enzyme needs no extension (`references/autodiff.md`).

## Outputs

`Base.get_extension(PatternVectors, :MooncakeExt) !== nothing` after `using Mooncake` (same for the others). Checked by
`test/test_Mooncake.jl`, `test/test_ReverseDiff.jl`, the sparse tests of `test/runtests.jl`.

## Verify (Kaimon)

| Check | Expect |
|:--|:--|
| `run_tests()` | `test_Mooncake.jl` ("Mooncake rule allocates per array, not per element" included) and `test_ReverseDiff.jl` pass |

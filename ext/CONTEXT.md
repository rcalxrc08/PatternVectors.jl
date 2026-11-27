# Extensions

One extension per weak dependency; the package loads and works without any of them.

## Inputs

| Load | Why |
|:--|:--|
| `MooncakeExt.jl` | the reverse rule of `materialize_mixture` (mixed broadcasts): `arg_role`, one fused NDual pass per differentiable argument, `Partial` |
| `ReactantExt.jl` | Reactant (XLA): a `PatternVector` reaches XLA as a fill, a concatenation of its bounds and an alternating pair (`traced_array`), pattern values behind an `optimization_barrier`, never as `n` values; traced indexing, `similar`/`collect` for traced pattern values (D-008, `references/reactant.md`) |
| `ReverseDiffExt.jl` | `TrackedStyle` wins over both PatternVector styles; `dot` with a `TrackedArray` recorded as ReverseDiff does |
| `SparseArraysExt.jl` | a PatternVector with a sparse vector promotes to the sparse style |
| `references/mooncake.md` | the design of the Mooncake rule, the alternatives evaluated, the cost model and the measurements |
| `references/reactant.md` | how a PatternVector reaches XLA, what Reactant and XLA fold, the measurements |
| `references/autodiff.md` | which backend needs what, upstream issues |
| `references/dependencies.md` | why each dependency is weak, compat floors |

## Process

- Mooncake: no rule for `PatternVector` itself (its derived tangent is structural); never `@from_rrule` the
  ChainRulesCore rules. `Partial` stays `@inline` with `Vararg{Any, M}` (a non-inlined vararg call boxes the elements:
  27 args took 18 s and 6 GB). `arg_role` decides both `_is_primitive` and the pullback; dispatching on the primal type
  broke `ZeroPattern` PatternVectors.
- ReverseDiff ≥ 1.18: `TrackedStyle` is an `AbstractArrayStyle`, so both argument orders must defer to it.
- Reactant: a new pattern gets a `traced_array` method built from fills, concatenations and broadcasts; the iota
  fallback is folded into a dense constant (`references/reactant.md`). Check with `test/reactant/runtests.jl`.
- Enzyme needs no extension (`references/autodiff.md`).

## Outputs

`Base.get_extension(PatternVectors, :MooncakeExt) !== nothing` after `using Mooncake` (same for the others). Checked by
`test/test_Mooncake.jl`, `test/test_ReverseDiff.jl`, the sparse tests of `test/runtests.jl`, `test/reactant/runtests.jl`
(local only).

## Verify (Kaimon)

| Check | Expect |
|:--|:--|
| `run_tests()` | `test_Mooncake.jl` ("Mooncake rule allocates per array, not per element" included) and `test_ReverseDiff.jl` pass |

# Stage 03: ChainRulesCore rules

The rules Zygote uses: the `PatternVector` constructor and `sum`.

## Inputs

| Load | Why |
|:--|:--|
| `rrules.jl` | `rrule(::Type{PatternVector}, n, pattern)` (through `pattern_to_vector_pullback`), `rrule(sum, ::AbstractVector)` |
| `references/autodiff.md` | the unthunk rule (#19), why the `sum` rule covers every vector (#20), Zygote versions |

## Process

- Every pullback unthunks its cotangent (Zygote ≥ 0.7.17 passes deferred `Thunk`s, #19).
- The `sum` rule applies to every `AbstractVector`, `CuArray`s included, on purpose (#20): an O(1), `isbits`
  `PatternVector(n, FillPattern(Δ))` cotangent. Do not restrict it to `PatternVector` without new measurements.
- These rules are not imported into Mooncake with `@from_rrule` (`references/mooncake.md`).

## Outputs

Zygote gradients accumulated on the pattern fields. Checked by `test/test_Zygote.jl` ("Zygote deferred cotangents"
keeps the cotangent of `sum` an `isbits` `PatternVector`).

## Verify (Kaimon)

The AD backends are test dependencies, not loadable from the project environment: use `run_tests()`.

| Check | Expect |
|:--|:--|
| `run_tests()` | `test_Zygote.jl` passes (its `@test_broken` cases stay broken) |

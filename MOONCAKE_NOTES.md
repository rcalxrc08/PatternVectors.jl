# Mooncake support — working notes

Status as of 2026-10-26, branch `support-mooncake`. Notes to resume the work in a later session.

## What is in place

- `src/rrules.jl` is the committed version (Zygote rules). The ChainRulesCore rrules must NOT be imported
  into Mooncake with `@from_rrule`: their pullbacks expect an array-like cotangent and would force a dense one.
- Mooncake needs no rule for `PatternVector` itself: its derived tangent is structural,
  `Tangent{(n = NoTangent, pattern = Tangent{pattern fields})}`, so gradients accumulate on the pattern fields.
  PatternVector-only gradients are O(1) in n (checked up to n = 10^18, 0 bytes allocated per call).
- `src/broadcast.jl`: the mixed broadcast (`PatternMixtureArrayStyle`) materializes pattern-only sub-expressions,
  flattens the rest (`Broadcast.flatten`, lazy, no allocation) and calls
  `materialize_mixture(style, f, args...)`, the single hook for AD rules.
- `ext/MooncakeExt.jl`: reverse rule for `materialize_mixture`.
  - Enabled per signature through `Mooncake._is_primitive` + `supports_fast_path(Sig)`; otherwise Mooncake's
    derived rule is used. Requirements: `DefaultArrayStyle`, singleton `f` with no tangent, args being
    `Array{<:IEEEFloat}`, IEEEFloat scalars, `PatternVector`s with all pattern fields of eltype T, or
    non-differentiable constants (incl. `Ref`s of them, e.g. `literal_pow`); one common float type;
    at most `MAX_DIRECTIONS = 16` directions; `f` must infer to `T` on primals and `NDual{T,K}` on duals.
  - Forward: plain broadcast only. Reverse: one fused loop re-evaluating `f` on `Mooncake.Nfwd.NDual`
    (one direction per dense/scalar arg, one per pattern field); dense cotangents accumulated in place
    (`newindex` handles broadcast dims), pattern/scalar cotangents in registers.
- `src/patternvector.jl`: `Core.kwcall` method forwarding `sum(x; dims = :)` to the positional,
  pattern-specialized `sum` (Zygote's `unbroadcast` calls the keyword form, which was O(n)).
  `ZeroPattern` got its O(1) `sum`.
- Tests: `test/runtests.jl` (core, no AD) includes `test/test_Zygote.jl` (Zygote + direct ChainRules rule calls)
  and `test/test_Mooncake.jl`. Line coverage is 100% with the
  CI settings (`--inline=yes`). The ChainRules pullbacks of pattern constructors are tested by calling `rrule`
  directly: through Zygote they get inlined and coverage does not count them.
  Benchmark: `test/bench_zygote_mooncake.jl` (kept local, not committed, not run by the tests).

## Benchmark summary (gradient time, Mooncake with prebuilt rule)

| case | n | Zygote | Mooncake |
|---|---|---|---|
| PV only, pattern chain | any n | ~12 µs (after the kwcall fix) | ~60 µs |
| `sum(pv .* x)` | 10^5 | 1.95 ms | 1.04 ms |
| `sum(h.(pv, x))`, `h(u,v) = sin(u)*exp(v) + u^2` | 10^5 | 7.6 ms | 11.2 ms |
| dense `sum(v .* x)` | 10^5 | 3.9 ms | 21 ms |

Mooncake allocates ~3x less than Zygote whenever arrays are involved.

## TODO — expensive broadcast functions (next step)

For expensive `f` (e.g. `h` above) the Mooncake rule is ~1.3x slower than Zygote: Zygote stores the
partials in the forward pass, our rule re-evaluates `f` with NDuals in the reverse pass (primal + dual pass).
Idea: optionally store the partials in the forward pass (`(K + 1) * n` extra floats), e.g. keep the
`Array{NDual{T,K}}` from a dual forward broadcast and extract the primal from it, choosing between the
two strategies by a heuristic or a user switch. Measure with `test/bench_zygote_mooncake.jl`.

## Other open points

- `rrule(::typeof(sum), x::AbstractArray{T,1})` in `src/rrules.jl` is type piracy (applies to every vector
  under Zygote); restricting it to `PatternVector` would be safer.
- Zygote mis-differentiates dense references like `gen.(1:n, n, a, b, c, d)` with `ifelse`
  (https://github.com/FluxML/Zygote.jl/issues/1602); `test/runbench.jl` uses that pattern.
- The phone (Galaxy S9+) connects via MTP: not a filesystem, cannot be used as a working disk.

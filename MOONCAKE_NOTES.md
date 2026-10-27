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
  - `f` is evaluated on `Mooncake.Nfwd.NDual` with ONE direction per differentiable argument (PatternVectors
    are seeded elementwise like dense arrays), so the dual width does not grow with the pattern size.
    The elementwise cotangent of a PatternVector is mapped onto the pattern fields with the pattern Jacobian
    (`getindex_pattern` on `seeded_pattern`, a pattern with NDual fields): cheap, no evaluation of `f`.
  - `PARTIALS_STRATEGY` (a `Ref` in the extension): `:store` (default) evaluates `f` once on NDuals in the
    forward pass and keeps K partials per element; `:recompute` runs the primal forward and re-evaluates `f`
    on NDuals in the pullback (no extra memory). Switch with
    `Base.get_extension(PatternVectors, :MooncakeExt).PARTIALS_STRATEGY[] = :recompute`.
  - Reverse pass: one fused loop; dense cotangents accumulated in place (`newindex` handles broadcast dims),
    scalar and pattern cotangents in registers. Helpers called inside `@inbounds` loops must be
    `@propagate_inbounds` (a missing one doubled the loop time).
  - GPU / non-Array arrays (e.g. CuArray): MooncakeCUDAExt's own broadcast rule rejects PatternVector args,
    so our rule must handle them. `uses_fused_loop` is false unless the style is `DefaultArrayStyle` and all
    dense args are `Array`; then the forward pass materializes the NDual broadcast (`:store`) and
    `broadcast_pullback` uses whole-array broadcasts/reductions, mapping PatternVector cotangents with the
    package's `pattern_to_vector_pullback` (GPU friendly). Tested on CPU with a custom broadcast style.
  - The plan (`mixture_plan`, needing `tangent_type`) is computed by the callers, which are `@assume_effects
    :foldable`, and passed to the `@generated` functions as `Val((roles, directions, K))`: generators run in
    their definition world and would not see `tangent_type(::CuArray)` from MooncakeCUDAExt (loaded later).
- `src/broadcast.jl`: `get_style` unwraps nested `PatternMixtureArrayStyle` (they appear when mixing with
  CuArrayStyle), so `materialize_mixture` always gets the innermost style.
- `test/cuda_tests.jl` (needs a GPU, not run by `runtests.jl`): Zygote and Mooncake on CuArrays. Run it in an
  environment with PatternVectors (dev), CUDA, Zygote, ChainRulesCore, BenchmarkTools, Mooncake, Statistics.
- `src/patternvector.jl`: `Core.kwcall` method forwarding `sum(x; dims = :)` to the positional,
  pattern-specialized `sum` (Zygote's `unbroadcast` calls the keyword form, which was O(n)).
  `ZeroPattern` got its O(1) `sum`.
- Tests: `test/runtests.jl` (core, no AD) includes `test/test_Zygote.jl` (Zygote + direct ChainRules rule calls)
  and `test/test_Mooncake.jl`. Line coverage is 100% with the
  CI settings (`--inline=yes`). The ChainRules pullbacks of pattern constructors are tested by calling `rrule`
  directly: through Zygote they get inlined and coverage does not count them.
  Benchmark: `test/bench_zygote_mooncake.jl` (kept local, not committed, not run by the tests).

## Benchmark summary (gradient time, Mooncake with prebuilt rule)

The laptop timings vary by up to ~30% between processes: compare variants in the same process.

| case | n | Zygote | Mooncake `:recompute` | Mooncake `:store` |
|---|---|---|---|---|
| PV only, pattern chain | any n | ~12 µs (after the kwcall fix) | ~60 µs | ~60 µs |
| `sum(pv .* x)` | 10^6 | 14.9 ms | 13.5 ms | 15.5 ms |
| `sum(h.(pv, x))`, `h(u,v) = sin(u)*exp(v) + u^2` | 10^6 | 57 ms | 63 ms | 45 ms |
| `sum(hh.(pv, x))`, heavier `hh` | 10^6 | 132 ms | 223 ms | 116 ms |
| 5 arguments, cheap | 10^6 | 103 ms | 56 ms | 60 ms |
| 5 arguments, heavy | 10^6 | 153 ms | 202 ms | 153 ms |
| dense `sum(v .* x)` (no PatternVector) | 10^5 | 3.9 ms | 21 ms | 21 ms |

Mooncake allocates less than Zygote whenever arrays are involved (`:store` adds K * n floats).

## Done: expensive broadcast functions

Was ~1.3x slower than Zygote. Fixed by (1) one dual direction per argument instead of per pattern field
(the main win: `h` went from ~14 ms to ~5 ms at n = 10^5) and (2) the `:store` strategy.
Possible follow-up: choose the strategy automatically (no static estimate of the cost of `f` is available).

## Enzyme (no extension needed)

Enzyme 0.13 differentiates PatternVector structurally, like Mooncake without its custom rule; tests in
`test/test_Enzyme.jl` (vs Zygote for scalar inputs, vs Mooncake for PatternVector gradients, n = 10^12 check).
- PatternVector is immutable: pass it as `Active`, not `Duplicated` (a `Duplicated` shadow silently stays zero).
  The returned gradient is a PatternVector with `n = 0` (Enzyme zeroes integer fields): read `.pattern`.
- Pattern gradients are O(1) in n (0.08 µs for `sum(p .* p)` at n = 10^3 ... 10^9).
- A constant dense array in a broadcast raises EnzymeRuntimeActivityError (same with plain Arrays): pass it
  `Duplicated` or use `set_runtime_activity(Reverse)`.
- n = 10^5: `sum(p .* x)` 0.52 ms (Mooncake 1.1 ms); `sum(h.(p, x))` with sin/exp 9.5 ms (Mooncake 3.8-6 ms),
  but Enzyme on the dense copy of `p` costs the same (8.9 ms): the gap is Enzyme's broadcast, not the patterns.
  A possible follow-up is an EnzymeRule for `materialize_mixture` reusing the NDual approach of MooncakeExt.
- CUDA: Enzyme fails on the MX150 even for dense CuArrays (kernel image load, CUDA error 801); with a
  PatternVector argument it fails earlier (`MixedDuplicated` MethodError in Enzyme's CUDA broadcast rule). Not
  verifiable on this machine.

## Other open points

- `rrule(::typeof(sum), x::AbstractArray{T,1})` in `src/rrules.jl` is type piracy (applies to every vector
  under Zygote); restricting it to `PatternVector` would be safer.
- Zygote mis-differentiates dense references like `gen.(1:n, n, a, b, c, d)` with `ifelse`
  (https://github.com/FluxML/Zygote.jl/issues/1602); `test/runbench.jl` uses that pattern.
- The phone (Galaxy S9+) connects via MTP: not a filesystem, cannot be used as a working disk.

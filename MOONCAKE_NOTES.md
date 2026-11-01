# Mooncake support — working notes

Status as of 2026-11-02: the rule described here is on `main`, on top of 0.1.5 (Mooncake support was added on
`support-mooncake`, the simplification on `mooncake-simplify`).
Notes to resume the work in a later session.

## What is in place

- `src/rrules.jl` is the committed version (Zygote rules). The ChainRulesCore rrules must NOT be imported
  into Mooncake with `@from_rrule`: their pullbacks expect an array-like cotangent and would force a dense one.
- Mooncake needs no rule for `PatternVector` itself: its derived tangent is structural,
  `Tangent{(n = NoTangent, pattern = Tangent{pattern fields})}`, so gradients accumulate on the pattern fields.
  PatternVector-only gradients are O(1) in n (checked up to n = 10^18, 0 bytes allocated per call).
- `src/broadcast.jl` (0.1.5): mixed and dense sub-expressions stay lazy and fuse into one call, scalar arithmetic
  is folded once, and the mixed broadcast (`PatternMixtureArrayStyle`) materializes pattern-only sub-expressions,
  flattens the rest (`Broadcast.flatten`, lazy, no allocation) and calls
  `materialize_mixture(style, f, args...)`, the single hook for AD rules.
- `ext/MooncakeExt.jl` (84 lines): reverse rule for `materialize_mixture`, one code path for every array
  type, written for cheap broadcasts such as Simpson weights (`h / 3 * sum(W .* y)`).
  - Enabled per signature through `Mooncake._is_primitive`; otherwise Mooncake's derived rule is used.
    Requirements: an `AbstractArrayStyle`, singleton `f` with no tangent, args being float arrays whose
    tangent type is the array itself (`Array`, `CuArray`), IEEEFloat scalars, `PatternVector`s with all pattern
    fields of eltype T, or non-differentiable constants (`Ref`s of them as in `literal_pow`, PatternVectors
    without differentiable fields such as Int8 Simpson weights or a ZeroPattern); one common float type; `f`
    must infer to `T` on primals and to `NDual{T,1}` when any one differentiable argument is seeded.
    Complex arrays are not supported (derived rule), e.g. the Int8 Simpson weights times complex vectors of
    FinancialFFT. Since 0.1.5 one complex array sends the whole fused expression to the derived rule
    (`a * real(s * y)`); gradients are correct. `real(s * exp(im * x * a))` still uses the rule.
  - Forward: the plain broadcast. Nothing is stored besides the output and its zero tangent.
  - Reverse: one fused pass per differentiable argument k, in which `f` is evaluated again with argument k
    seeded as `Mooncake.Nfwd.NDual{T,1}` (`Partial{k}`, k a type parameter). Dense args: `dx .+= dy .* ∂y/∂x`,
    summed over broadcast dimensions; scalars: one lazy sum; PatternVectors: one reduction against a
    PatternVector whose pattern fields are NDuals seeded per field, so `getindex_pattern` supplies the pattern
    Jacobian for any pattern. Broadcasts and reductions only: CPU and GPU share the code.
  - `arg_role` (`@assume_effects :foldable`: constant at compile time, and it sees `tangent_type` methods of
    extensions loaded later, e.g. CuArray in MooncakeCUDAExt) decides both `_is_primitive` and the pullback
    (`cotangent!(Val(role), ...)`). Dispatching on the primal type instead broke ZeroPattern PatternVectors with
    Float64 eltype: they have no tangent but matched the PatternVector method (test added).
  - PatternVector is immutable: its cotangent is rdata, built as a PatternVector holding the field cotangents
    and converted with `Mooncake.primal_to_tangent!!`.
  - `Partial` must stay `@inline` with `Vararg{Any, M}`: since 0.1.5 fused expressions are large, and a call
    through varargs that is not inlined boxes the elements. A long expression (27 args) took 18 s and 6 GB until
    this was fixed; the test "Mooncake rule allocates per array, not per element" guards it.
  - Cost model: `f` is evaluated 1 + K times (K = differentiable arguments, counting every occurrence, because
    `Broadcast.flatten` does not deduplicate: 21 of the 27 args of the long expression) and no partials are stored. The
    earlier versions evaluated `f` once on `NDual{T,K}` and stored K partials per element. Cheap `f` (products,
    sums, Simpson weights) got faster; expensive `f` with several differentiable arguments got slower (below).
  - History: 2026-10-27 fused generated CPU loop + separate GPU path + store/recompute switch (248 lines);
    2026-10-28 store `NDual{T,K}` output + K broadcast passes (111 lines, slower than the first for cheap f);
    2026-10-29 this version.
  - Evaluated and not used: `ChainRulesCore.ProjectTo` (Mooncake never calls it, and it returns a natural
    cotangent while Mooncake needs the structural one); a custom Mooncake tangent type (documented for
    recursive types; it needs `lgetfield`/`_new_` rules and the whole tangent interface, i.e. more code);
    `pattern_to_vector_pullback` in the rule (one pass per pattern field, slower than the Jacobian reduction).
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
  `test/test_Mooncake.jl` also runs the rule with a custom broadcast style (`GenericArrayStyle`), which exercises
  the non-default-style (GPU-like) code on CPU, and covers constant Int8 Simpson weights and ZeroPattern
  PatternVectors in mixed broadcasts. Mooncake's `TestUtils.test_rule` (reverse mode, `unsafe_perturb = true`)
  also passes on the rule for these cases (checked by hand, not in the test suite).
  Benchmark: `test/bench_zygote_mooncake.jl` (kept local, not committed, not run by the tests).

## Benchmark summary (gradient time, Mooncake with prebuilt rule)

Timings vary by up to 2x between processes while other Julia processes run (another session was active):
compare variants in one process, interleaved. 2026-10-29, load about 10%, minimum over runs: the three versions
of the rule were loaded side by side under different hook names (scratch script, not committed); Zygote
differentiates the dot broadcast of the same function. `W` = constant Int8 Simpson weights, `w` = PaddedEvenOdd
PatternVector of 4 differentiable scalars, K = number of differentiable arguments of the broadcast.

| case | K | n | Zygote | fused loop (10-27) | store (10-28) | current |
|---|---|---|---|---|---|---|
| `sum(w .* x)` | 2 | 10^3 | 9.2 µs | 6.7 µs | 8.8 µs | 6.2 µs |
| | | 10^5 | 629 µs | 1.12 ms | 1.82 ms | 950 µs |
| | | 10^6 | 14.5 ms | 14.3 ms | 21.1 ms | 11.0 ms |
| `h / 3 * sum(W .* y)` | 1 | 10^5 | 212 µs | 438 µs | 931 µs | 399 µs |
| | | 10^6 | 5.3 ms | 10.8 ms | 14.4 ms | 8.5 ms |
| `h / 3 * sum(W .* exp.(-x .^ 2))` | 1 | 10^3 | 29.8 µs | 11.6 µs | 11.9 µs | 22.8 µs |
| | | 10^6 | 26.9 ms | 18.4 ms | 21.5 ms | 27.5 ms |
| `sum(w .* exp.(-x .^ 2))` | 2 | 10^6 | 42.8 ms | 27.6 ms | 36.7 ms | 41.1 ms |
| `sum(h.(w, x))`, `h(u,v) = sin(u)*exp(v) + u^2` | 2 | 10^6 | 55.1 ms | 41.5 ms | 49.5 ms | 70.4 ms |
| `sum(hh.(w, x, a))`, heavier `hh` | 3 | 10^6 | 104 ms | 87.6 ms | 96.5 ms | 230 ms |
| `sum(@. p * x + q * x * d - a)`, two PatternVectors | 5 | 10^3 | 79.6 µs | 40.9 µs | 44.9 µs | 8.4 µs |
| | | 10^6 | 86.1 ms | 51.6 ms | 70.8 ms | 13.4 ms |
| `sum(X .* w .+ a)`, X of size 10^4 x 100 | 3 | 10^6 | (30.6 ms) | 18.2 ms | 31.9 ms | 17.6 ms |

(The Zygote time of the last row goes through ForwardDiff duals; its own rules for `X .* w .+ a` take ~13 ms.)
Plain Simpson case at n = 10^6, current version (8.9 ms): rule forward `W .* y` 2.7 ms (mostly allocating the
output), zeroed output tangent 1.9 ms, rule pullback 1.2 ms; Mooncake's own zeroed input tangent 1.9 ms and `sum`
with its pullback 0.75 ms. Zygote (5.9 ms) materializes no output tangent (its `sum` pullback is a lazy `Fill`).
GPU (MX150, Float32, same process, interleaved; `test/cuda_tests.jl` passes), fused loop / store / current at
n = 10^6: `sum(w .* x)` 4.02 / 2.56 / 1.33 ms, `h / 3 * sum(W .* y)` 1.70 / 1.75 / 1.07 ms,
`h / 3 * sum(W .* exp.(-x .^ 2))` 1.70 / 1.74 / 1.20 ms, `sum(h.(w, x))` 4.51 / 2.99 / 1.59 ms, `sum(X .* w)`
3.50 / 2.83 / 1.79 ms; at n = 10^5 the current version is also the fastest (`sum(w .* x)` 0.98 / 0.55 / 0.45 ms).
On the GPU recomputing is cheaper than storing partials even for `h`. The fused-loop version used
`pattern_to_vector_pullback` on the GPU.
PatternVector-only code does not use the rule (O(1) in n, ~60 µs in Mooncake vs ~12 µs in Zygote).

## Trade-off: expensive broadcast functions

The earlier versions evaluated `f` once on `NDual{T,K}` (one direction per argument rather than per pattern field:
that took `h` from ~14 ms to ~5 ms at n = 10^5) and stored the partials. The current one evaluates `f` once per
differentiable argument plus once forward: with K >= 2 and transcendental functions it is 1.5x (`w .* exp`) to
2.6x (`hh`, K = 3) slower than the fused loop. If this matters, a store path (forward on `NDual{T,K}`) could be
chosen for such signatures, at the cost of the per-argument direction bookkeeping removed here. This is a CPU
trade-off: on the GPU the current version is the fastest in every measured case, `h` included.

Main 0.1.5 did not make any line of the extension redundant, but it changes what `f` is. Nested expressions now
reach the rule as one fused call, so the cost model applies to the whole expression instead of to each step
(gradient ms at n = 10^6, about 16-20% load; eager = rule on 0.1.4, fused = rule on 0.1.5):

| case | eager | fused | Zygote |
|---|---|---|---|
| `a * (w * x + d)` | 22.05 | 16.23 | 25.29 |
| `(w * x)^2` | 25.44 | 18.96 | 39.87 |
| `a * (b * (w * x) + c)` | 34.30 | 16.78 | 26.67 |
| `a * sin(w * x)` | 77.24 | 84.70 | 54.54 |
| `sin(w * x) * exp(w)` | 66.59 | 80.17 | 55.17 |

Memory also drops (39 to 23 MB, three-level 70 to 23 MB): intermediates no longer get an output tangent.
Scalar folding lowers K only slightly (`(a + b) * w * x`).

Pullback versions on the real fused expressions (2026-11-01, load 12-19%, gradients agree). `main` is the committed
fused-loop rule of 0.1.5, `main + inl` the same with `@inline` on `SeededFunction` (never committed: the current rule
replaced that code). `main + inl` and current on the rows with `^2`, the `sin` row and the long ones come from three alternating
runs of real package copies (minimum); everything else is one process, interleaved. Hand-written `f` hide the boxing: real expressions go through
`Broadcast.flatten` closures, so compare with those.

| case | n | K | main | main + inl | current | Zygote |
|---|---|---|---|---|---|---|
| Simpson `W * x` | 10^6 | 1 | 13.01 ms | 12.59 ms | 10.58 ms | 14.64 ms |
| `W * exp(-x^2)` | 10^6 | 1 | 2.23 s | 23.88 ms | 34.46 ms | 53.93 ms |
| `w * x` | 10^6 | 2 | 20.44 ms | 20.52 ms | 15.89 ms | 17.54 ms |
| `p * x + q * x * d - a` | 10^6 | 6 | 8.56 s | 35.45 ms | 18.88 ms | 63.87 ms |
| `a * (w * x + d)` | 10^6 | 4 | 4.87 s | 32.68 ms | 20.28 ms | 29.16 ms |
| `(w * x)^2` | 10^6 | 2 | 2.37 s | 19.64 ms | 16.71 ms | 34.98 ms |
| `a * (b * (w * x) + c)` | 10^6 | 5 | 6.73 s | 31.74 ms | 20.77 ms | 36.06 ms |
| `a * sin(w * x)` | 10^6 | 3 | 2.24 s | 39.23 ms | 83.97 ms | 58.38 ms |
| `sin(w * x) * exp(w)` | 10^6 | 3 | 2.08 s | 46.18 ms | 101.19 ms | 66.07 ms |
| long, 11 args | 10^5 | 9 | 1.18 s | 9.95 ms | 28.57 ms | 21.12 ms |
| long, 25 args | 10^5 | 21 | 107 ms (derived) | 33.75 ms (derived) | 94.30 ms | 59.50 ms |

Memory at n = 10^6: current 30.5 MB everywhere, `main + inl` 38-76 MB, `main` up to 3.3 GB. The 25-arg row exceeds
`MAX_DIRECTIONS = 16` of the committed rule, so it runs Mooncake's derived rule, 2.8x faster than the current rule:
above some K the current rule loses to the fallback. A cap on K in `_is_primitive` would take one condition; the
crossover is not measured (K = 9 is not covered).
A 27-arg expression at n = 10^5 went from 36-42 ms (34 MB) before 0.1.5 to 78-111 ms (3.1 MB) after it, Zygote 25-38 ms.
Not implemented: deduplicating repeated arguments (scalars and immutable PatternVectors cannot be identified by `===`).

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
- `Project.toml` allows `Mooncake = "0.5"`, but `Mooncake.Nfwd` (used by the extension) exists only since
  0.5.25: the lower bound should be raised.
- `pattern_to_vector_pullback` (Zygote's PatternVector constructor pullback) allocates a dense vector per
  pattern field for a dense cotangent (`sum(Δapv .* indicator)`). A lazy sum with the wrapped broadcast style
  removes the allocations (tried and measured on 2026-10-28, not committed).
- Zygote mis-differentiates dense references like `gen.(1:n, n, a, b, c, d)` with `ifelse`
  (https://github.com/FluxML/Zygote.jl/issues/1602); `test/runbench.jl` uses that pattern.
- The phone (Galaxy S9+) connects via MTP: not a filesystem, cannot be used as a working disk.
- The README performance table (`etc/benchmarks.jl`, n = 10^5) was regenerated on 2026-11-02 at about 16% load: every
  column, dense included, reads about 40% slower than the previous table, so compare within the table. On `sum(sin.(w .* x) .* exp.(w))` Mooncake is now behind Zygote
  and Enzyme: the fused expression is recomputed per argument.

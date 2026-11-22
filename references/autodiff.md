# Automatic differentiation

_What each backend needs, the rules of the package and the upstream issues behind them._ (Layer 3 reference.) The
Mooncake rule has its own reference, `mooncake.md`.

## Zygote (ChainRulesCore rules, `src/03_rrules/`, `src/04_patterns/`)

- Every pullback (these and the pattern constructors' in `src/04_patterns/`) unthunks its cotangent: Zygote ≥ 0.7.17
  passes the cotangent of a scalar as a deferred `Thunk`, and a `FillPattern` of a thunk became, through Zygote's `map`
  in `unthunk_tangent`, a host `Vector` cotangent of `sum`, which broke GPU gradients (FinancialFFT's COS, densities,
  Carr-Madan on `CuArray`s; #19). The test "Zygote deferred cotangents" checks that the cotangent of `sum` stays an
  isbits `PatternVector`; `test/Project.toml` requires Zygote 0.7.18.
- The `rrule` of `sum` applies to every vector, `CuArray`s included, on purpose (#20): its
  `PatternVector(n, FillPattern(Δ))` cotangent is O(1) and isbits, where Zygote's own rule materializes a full array,
  and `FillArrays.Fill` is materialized by broadcasts of 3-argument functions (`muladd`, `ifelse`). Measured (docs
  `ad.md`, n = 10^6, Zygote 0.7.18): gradients up to 8x faster on the CPU, up to 2x on the GPU, and less memory in every
  case; keep it. (It is type piracy on `AbstractVector`; restricting it was considered and rejected by #20.)
- The keyword `sum(x; dims = :)` that Zygote's `unbroadcast` calls is forwarded to the pattern `sum` (`Core.kwcall` in
  `src/01_vector/patternvector.jl`); it was O(n).
- The ChainRules pullbacks of pattern constructors are tested by calling `rrule` directly: through Zygote they get
  inlined and coverage does not count them.
- Zygote mis-differentiates dense references like `gen.(1:n, n, a, b, c, d)` with `ifelse` (FluxML/Zygote.jl#1602);
  some Zygote cases are pinned with `@test_broken`.

## `dot` (`src/05_linearalgebra/`)

- `dot` uses broadcasts and a reduction only, so it runs on the device of a dense argument (the generic `dot` indexes
  element by element). ChainRules reaches it in reverse mode: the `rrule` of `sum` returns a `PatternVector`
  cotangent, and the cotangent of a scalar times an array is `dot(Ȳ, B)` (#4).
- `dot` has its own `rrule`s (`dot_rrule`, #21), for the weighted sums of quadratures `dot(w, y)` with constant weights:
  no `w .* y` in the forward pass, the cotangent of the dense argument a `PatternVector`, none for integer or Bool
  weights. ChainRules' generic rule for two arrays gave no gain over `sum(w .* y)`; this one allocates 1.3-1.75x less
  on the GPU and runs 1.2-1.5x faster there (n = 10^6), about the same on the CPU.
- In FinancialFFT (Lewis) and VibratoMonteCarlo (trapezoid) switching `sum(w .* y)` to `dot` was measured neutral (±4%,
  same memory; Lewis's 10^4-node sum is small next to the characteristic exponent, and Vibrato does not use Zygote),
  so they keep `sum`.

## Enzyme (no extension)

- Enzyme 0.13 differentiates `PatternVector` structurally. `PatternVector` is immutable: pass it as `Active`, not
  `Duplicated` (a `Duplicated` shadow silently stays zero). The returned gradient is a `PatternVector` with `n = 0`
  (Enzyme zeroes integer fields): read `.pattern`. Pattern gradients are O(1) in n.
- A constant dense array in a broadcast raises `EnzymeRuntimeActivityError` (same with plain `Array`s): pass it
  `Duplicated` or use `set_runtime_activity(Reverse)`.
- Enzyme ≥ 0.13.219 (Enzyme_jll 0.0.307) aborted with an LLVM assertion (`cast<Instruction>` of the shadow of a known
  allocation, in `handleKnownCallDerivatives`) on `sum(p .* p .+ 2)`: the `PatternVector` constructor's length check,
  inlined into the `materialize` of every broadcast, allocated in its error branch (boxed `n`, message string,
  `DomainError`). The throw is now out of line (`@noinline throw_length_error`), and the cap of 0.13.218 is gone
  (EnzymeAD/Enzyme.jl#4053, #15). Keep error paths of functions that run inside differentiated code out of line, for
  the same reason. Enzyme 0.13.221 fixed it upstream: the MWE passes even on 7139448 (the inline throw), Julia 1.12.7
  and 1.13.4, plain `Reverse` (commented on #4053 on 2026-11-19, left open for the maintainers); the out-of-line throw
  stays as good practice.
- CUDA: Enzyme failed on the MX150 even for dense `CuArray`s (kernel image load, CUDA error 801); with a
  `PatternVector` argument it fails earlier (`MixedDuplicated` MethodError in Enzyme's CUDA broadcast rule).
- A possible follow-up is an EnzymeRule for `materialize_mixture` reusing the NDual approach of `MooncakeExt`
  (`mooncake.md`, "Enzyme").

## ReverseDiff (`ext/ReverseDiffExt.jl`)

Since ReverseDiff 1.18 `TrackedStyle` is an `AbstractArrayStyle`, and both packages claim precedence over every
`AbstractArrayStyle`: both argument orders defer to the ReverseDiff rule, which then broadcasts over the values.
`dot` with a `TrackedArray` is recorded as ReverseDiff does.

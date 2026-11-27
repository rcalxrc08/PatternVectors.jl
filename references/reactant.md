# Reactant (XLA)

_How a PatternVector reaches an XLA program, and what was measured._ (Layer 3 reference; `ext/ReactantExt.jl`, #24.)

## What Reactant does with an array it does not know

Reactant traces a function into a StableHLO program. An argument of a broadcast that is not a traced array goes
through `Reactant.broadcast_to_size`, whose `AbstractArray` method converts it into a dense constant: without the
extension, `w .* exp.(x)` embeds the `n` values of `w` in the program (`stablehlo.constant dense<...> : tensor<n>`).
Reactant's other operations go through `ReactantCore.materialize_traced_array`, whose default returns the array as it is,
and its indexing of an `AbstractArray{<:TracedRNumber}` follows `parent` (infinite recursion on a PatternVector).

## The lowering (`traced_array`)

Each pattern is built from the structural operations of XLA, from its values only:

| Pattern | Operations |
|:--|:--|
| `FillPattern`, `ZeroPattern` | broadcast of the scalar |
| `InitialValuePattern`, `FinalValuePattern`, `PaddedFillPattern` | concatenation of the bounds (`tensor<1>`) and a broadcast |
| `EvenOddPattern` | the pair `[a, b]` broadcast to `2 × ⌈m/2⌉`, reshaped, sliced to `m` |
| `PaddedEvenOddPattern` | concatenation of the bounds and the alternating inner part |
| any other pattern | `getindex_pattern` on an iota (fallback, folded: see below) |

Pattern values that are numbers go through `Ops.optimization_barrier` (scalars only); traced values do not need it.

## What gets folded into a dense constant (measured, Reactant 0.2.309, 2026-11-27)

- Reactant's own optimization passes fold every constant expression with at most 1024 elements
  (`max_constant_threshold`): without the barrier, every pattern with constant values became a dense constant for
  `n ≤ 1024`.
- XLA folds the concatenation of a constant and a broadcast of a constant (PaddedFill: a constant of `n` values at
  `n = 10^6`) and the comparisons of an iota with constants (a `pred[n]` mask); it never folds a broadcast, nor the
  reshape of one (EvenOdd stayed `f64[2]` broadcast and bitcast).
- With the barrier on the scalars, no constant larger than 2 elements is left at any `n`, in StableHLO and in the
  optimized XLA program, for every pattern (`test/reactant/runtests.jl` checks it at `n = 2001`).

## Measurements: `sum(w .* exp.(x))`, PatternVector against `collect(w)` captured as a constant

Compile time (`@compile`), i7-8550U, Windows CPU and WSL2 CUDA (MX150):

| n | PatternVector | dense constant |
|--:|--:|--:|
| 10^5 | 0.15-0.2 s | 1.0-1.2 s |
| 10^6 | 0.15-0.6 s | 4-10 s |
| 4·10^6 | 0.6 s | 15-35 s |

The dense constant costs compile time and program size linearly in `n`; the PatternVector a constant.

Run time at `n = 10^6`:

- CUDA (MX150): equal within the noise (PatternVector 3.8-4.2 ms, dense 4.2-6.3 ms; 4·10^6: 5.2-7.5 ms against
  5.5-11 ms), the structural operations fused into the reduction.
- XLA CPU: Fill as fast as dense (0.9 against 1.1 ms); PaddedFill and PaddedEvenOdd 2.5 ms, EvenOdd 5.2 ms. XLA CPU
  sends the multiply-exp-reduce to a YNNPACK fusion that takes no concatenation, so the weights are computed by a
  separate loop fusion into a buffer of `n`, read back by the reduction; the EvenOdd broadcast-bitcast does stay
  inside the YNNPACK fusion but runs slower there. A dense constant is a buffer read once.

## Rejected alternatives

- `getindex_pattern` on an iota for every pattern: compact to write, but folded (above) and, where not folded, a
  separate fusion with integer arithmetic for `isodd`; kept only as the fallback for patterns without a method.
- Dense constant (Reactant's default): loses the structure, compile time linear in `n`.

# Mission

_Why the package exists and how to tell whether it succeeds._ (Layer 3 reference: stable, changes only with a decision
in `decisions.md`.)

## Purpose

Quadrature weights, padded and alternating vectors repeat a few values over long lengths. Storing them densely costs
memory and, under AD, a dense cotangent per weight vector. `PatternVector` stores the length and the distinct values,
evaluates broadcasts once per distinct value, and lets every AD backend accumulate gradients on those values, on the CPU
and on the GPU.

## Goals

- Pattern-only expressions cost O(1) in the length, forward and reverse (gradients checked up to n = 10^18).
- Mixed expressions with dense arrays fuse into one loop on the dense array's device, never slower than the dense
  equivalent.
- Gradients with Zygote, Mooncake, Enzyme and ReverseDiff, on `Array`s and `CuArray`s.

## Constraints

- Any number type (Float16, Float32, Float64, irrationals, duals, complex); any array type with a broadcast style.
- A `PatternVector` of `isbits` values is `isbits`.
- Light hard dependencies: ChainRulesCore and LinearAlgebra; backends are weak dependencies.

## Success criteria

- `test/runtests.jl` (memory bounds of nested broadcasts included) and the AD test files pass on CI (1.11, 1.12).
- The `docs/src/performance.md` table: PatternVector at least as fast as dense on every row.

## Non-goals

- Multidimensional arrays (FillArrays.jl covers constant arrays of any dimension).
- Lazy concatenations in general (LazyArrays.jl).
- Mutable vectors: a `PatternVector` is immutable; in-place broadcasts into it are not supported.

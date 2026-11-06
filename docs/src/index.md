```@raw html
<img class="pkg-hero-logo" src="assets/logo.png" alt="PatternVectors.jl logo" width="96" height="96">
```

# PatternVectors.jl

```@raw html
<p class="pkg-roles"><b>Structured vectors</b> · <b>GPU</b> · <b>Automatic differentiation</b></p>
<div class="pkg-intro">
<p>An immutable <code>AbstractVector</code> that stores a few values and the rule for where they go, instead of every
element: Simpson weights <code>[1, 4, 2, 4, …, 4, 1]</code>, alternating signs or boundary values on a grid. A tiny
<code>isbits</code> object, with <code>O(1)</code> memory, that broadcasts, sums and runs on the GPU, and that Zygote,
Mooncake and Enzyme differentiate.</p>
</div>
<nav class="pkg-links">
<a class="pkg-btn" href="https://github.com/rcalxrc08/PatternVectors.jl"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg> GitHub</a>
<a class="pkg-btn" href="https://github.com/rcalxrc08/PatternVectors.jl/stargazers">★ Star</a>
<a class="pkg-btn" href="#Quick-start">Get started ↓</a>
<a class="pkg-btn" href="https://rcalxrc08.github.io/">All projects</a>
</nav>
```

## Why

Numerical code is full of long vectors that hold only a few distinct values: quadrature weights
(`[1, 4, 2, 4, 2, …, 4, 1] / 3` for Simpson's rule), alternating signs, boundary values on a grid, a payoff that is
different only at maturity. The usual options both have a cost:

- **materializing a dense vector** uses O(n) memory, and every broadcast over it evaluates the function n times;
- **writing an index loop** (`ifelse(isodd(i), …)`) is fast on the CPU, but it does not run on GPUs without a kernel,
  and it is awkward for automatic differentiation.

A [`PatternVector`](@ref) keeps the vectorized style and removes the cost:

- it stores only its length and the pattern values, so it is a small `isbits` object that can be sent to a GPU kernel
  as is;
- a broadcast over `PatternVector`s evaluates the function once per distinct value and returns a `PatternVector`;
- a broadcast mixing `PatternVector`s and other arrays (`Array`, `CuArray`, …) returns an array of the other type,
  computed in a single fused pass, on CPU and GPU alike;
- `sum` and indexing with ranges are O(1) and return the expected types;
- it is differentiable with Zygote, Mooncake, Enzyme and ReverseDiff, see [Automatic differentiation](ad.md).

## Available patterns

| Pattern | Elements | Minimum length |
|:--|:--|:--|
| [`ZeroPattern(x)`](@ref ZeroPattern) | `[0, 0, 0, 0, …]` (zero of the type of `x`) | 1 |
| [`FillPattern(a)`](@ref FillPattern) | `[a, a, a, a, …]` | 1 |
| [`InitialValuePattern(a, b)`](@ref InitialValuePattern) | `[a, b, b, b, …]` | 2 |
| [`FinalValuePattern(a, b)`](@ref FinalValuePattern) | `[a, a, a, …, a, b]` | 2 |
| [`PaddedFillPattern(x, a, y)`](@ref PaddedFillPattern) | `[x, a, a, …, a, y]` | 3 |
| [`EvenOddPattern(a, b)`](@ref EvenOddPattern) | `[a, b, a, b, …]` (`a` at odd, `b` at even indices) | 2 |
| [`PaddedEvenOddPattern(x, a, b, y)`](@ref PaddedEvenOddPattern) | `[x, a, b, a, b, …, y]` (`a` at even, `b` at odd inner indices) | 4 |

A vector is built from a length and a pattern, `PatternVector(n, pattern)`. New patterns can be defined outside the
package, see [New patterns](patterns.md).

## Installation

PatternVectors.jl is registered in [LocalRegistry](https://github.com/rcalxrc08/LocalRegistry):

```julia
using Pkg
Pkg.Registry.add(Pkg.RegistrySpec(url = "https://github.com/rcalxrc08/LocalRegistry.git"))
Pkg.add("PatternVectors")
```

## Quick start

```@example quickstart
using PatternVectors
x = PatternVector(10, EvenOddPattern(0.2, 2.3))
z = PatternVector(10, PaddedEvenOddPattern(0.2, -2.0, 4.0, 2.3))
y = randn(10)
sin.(x)              # sin is evaluated twice
```

```@example quickstart
@. sin(x) + exp(z)   # the patterns are combined: still a PatternVector
```

```@example quickstart
@. sin(x) * y + z    # mixed with a Vector: a Vector, in one fused loop
```

Indexing with a range keeps the pattern, and `sum` is O(1):

```@example quickstart
z[2:2:8], sum(z)
```

More examples, Simpson's rule among them, are in [Getting started](usage.md).

## GPU

Being `isbits`, a `PatternVector` mixes with GPU arrays without any specialization of the code and without scalar
indexing:

```julia
using PatternVectors, CUDA
CUDA.allowscalar(false)
x = PatternVector(10, EvenOddPattern(0.2f0, 2.3f0))
z = PatternVector(10, PaddedEvenOddPattern(0.2f0, -2.0f0, 4.0f0, 2.3f0))
y = cu(randn(Float32, 10))
@. sin(x) * y + z   # CuArray
```

## Similar packages

- [FillArrays.jl](https://github.com/JuliaArrays/FillArrays.jl): lazy constant arrays (`Fill`, `Zeros`, `Ones`) of any
  dimension, the multidimensional counterpart of `FillPattern` and `ZeroPattern`.
- [LazyArrays.jl](https://github.com/JuliaArrays/LazyArrays.jl): lazy concatenations and broadcasts; a padded vector
  can be written as `Vcat(x, Fill(a, n - 2), y)`, though broadcasts over it are not reduced to the distinct values and
  alternating layouts are not covered.
- [SparseArrays](https://github.com/JuliaSparse/SparseArrays.jl): vectors that are zero except at a few indices.
- [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl): `isbits` arrays of small, fixed size, for the
  opposite case (few elements, all different).

PatternVectors.jl targets structured vectors of arbitrary length, with broadcasting, GPU and AD support built in.

## Contents

```@contents
Pages = ["usage.md", "ad.md", "performance.md", "patterns.md", "api.md"]
Depth = 2
```

# Getting started

## Building a PatternVector

A [`PatternVector`](@ref) is built from a length and a pattern. All the values of a pattern must have the same type,
and the length must be at least the minimum length of the pattern.

```@example usage
using PatternVectors
x = PatternVector(7, EvenOddPattern(0.2, 2.3))
```

```@example usage
initial_value, value_even, value_odd, final_value = 0.2, 2.3, -0.2, 1.3
PatternVector(7, PaddedEvenOddPattern(initial_value, value_even, value_odd, final_value))
```

## Indexing

A `PatternVector` is closed under indexing with ranges:

```@example usage
apv = PatternVector(70, PaddedEvenOddPattern(0.2, -2.0, 4.0, 2.3))
apv[1:7:50]
```

## Broadcasting

A scalar function broadcast over a `PatternVector` is evaluated once per distinct value and returns a
`PatternVector`:

```@example usage
av = PatternVector(10, EvenOddPattern(0.2, -2.0))
@. sin(av)
```

Broadcasting between `PatternVector`s returns a `PatternVector`; broadcasting a `PatternVector` with **any** other
`AbstractArray` returns an array of the other type, computed in one fused loop:

```@example usage
x = PatternVector(10, EvenOddPattern(0.2, 2.3))
y = randn(10)
z = PatternVector(10, PaddedEvenOddPattern(0.2, -2.0, 4.0, 2.3))
@. sin(x) * y + z
```

In a mixed broadcast, a sub-expression on `PatternVector`s only is still evaluated on the patterns: in
`exp.(x) .* y`, `exp` is called twice, not ten times.

## Simpson's rule

Quadrature weights without index loops, on CPU and GPU alike:

```@example usage
simpson(f_x, h) = h * sum(PatternVector(length(f_x), PaddedEvenOddPattern(1/3, 4/3, 2/3, 1/3)) .* f_x)
h = 0.01
simpson(exp.(0:h:1), h) - (exp(1) - 1)
```

`dot(w, f_x)` from LinearAlgebra gives the same result with broadcasts and a reduction only, so that it runs on the
device of `f_x`.

## Testing

```julia
using Pkg
Pkg.test("PatternVectors")
```

The tests cover the patterns and the broadcasts, then Zygote, ReverseDiff, Mooncake and Enzyme.

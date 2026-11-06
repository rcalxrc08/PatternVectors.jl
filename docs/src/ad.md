# Automatic differentiation

| Backend | Support | Gradient with respect to a `PatternVector` |
|:--|:--|:--|
| [Zygote.jl](https://github.com/FluxML/Zygote.jl) | ChainRules rules shipped with the package | a `PatternVector` for expressions involving only patterns, a dense vector when mixed with dense arrays |
| [Mooncake.jl](https://github.com/chalk-lab/Mooncake.jl) | package extension, with a dedicated rule for broadcasts mixing `PatternVector`s and arrays (CPU and CUDA) | a structural tangent holding the gradient of the pattern fields |
| [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl) | package extension (ReverseDiff 1.18 or later), for broadcasts mixing `PatternVector`s with tracked values | not available: a `PatternVector` enters as a constant |
| [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) | works natively, no extension needed | pass the vector as `Active`: the result is a `PatternVector` whose `.pattern` holds the gradient |

With Mooncake and Enzyme the gradient with respect to a `PatternVector` is accumulated directly on its pattern values,
never on a dense vector, so its cost does not depend on the length.

## Zygote

```@example ad
using PatternVectors, Zygote
f(a, b, x) = sum(sin.(PatternVector(length(x), EvenOddPattern(a, b))) .* x)
Zygote.gradient(f, 0.3, 0.5, rand(100))
```

An expression on `PatternVector`s only is differentiated on the patterns, whatever the length:

```@example ad
v = PatternVector(10^12, PaddedEvenOddPattern(1.0, 2.0, 3.0, 4.0))
Zygote.gradient(v -> sum(v .* v), v)
```

## Enzyme

A `PatternVector` is an immutable struct: pass it as `Active` (a `Duplicated` shadow stays zero), and pass constant
dense arrays as `Duplicated` or enable `set_runtime_activity`, as for any broadcast in Enzyme.

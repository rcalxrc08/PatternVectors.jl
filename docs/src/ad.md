```@raw html
<div class="fc-page-head">
<p class="fc-kicker"><span>Guide 02</span> Gradients on the pattern values</p>
```

# Automatic differentiation


```@raw html
<div class="fc-meta"><span>Zygote</span><span>Mooncake</span><span>Enzyme</span><span>ReverseDiff</span></div>
<nav class="fc-toc">
<a href="#Zygote"><b>01</b>Zygote</a>
<a href="#Enzyme"><b>02</b>Enzyme</a>
</nav>
</div>
```
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

### The cotangent of `sum`

The ChainRules rule of `sum` that ships with the package applies to every vector, dense arrays and `CuArray`s
included: the cotangent of `sum(x)` is `PatternVector(length(x), FillPattern(Δ))`. It is O(1) in memory and `isbits`,
so it enters CPU loops and GPU kernels as it is. Zygote's own rule materializes a full array of the length of `x`,
and `FillArrays.Fill`, the usual lazy alternative, is materialized by broadcasts of functions of three arguments
(`muladd`, `ifelse`), which a `PatternVector` evaluates once.

```@example ad
x = rand(5)
Zygote.gradient(x -> sum(x) * 2, x)[1]
```

Zygote gradients with Zygote 0.7.18, n = 10^6, Float64 on the CPU and Float32 on a GPU (NVIDIA MX150), minimum time
and memory allocated, this rule against Zygote's own:

| gradient of | CPU | CPU memory | GPU | GPU memory |
|:--|--:|--:|--:|--:|
| `sum(x)` | 0.47 ms vs 3.3 ms | 0.6 KB vs 7.6 MB | 0.16 ms vs 0.32 ms | 1 KB vs 3.8 MB |
| `sum(x .* w)` | 12.2 ms vs 15.2 ms | 22.9 MB vs 30.5 MB | 0.86 ms vs 1.17 ms | 11.4 MB vs 15.3 MB |
| `sum(exp.(x .* w))` | 26.3 ms vs 30.3 ms | 38.1 MB vs 45.8 MB | 1.19 ms vs 1.48 ms | 11.4 MB vs 15.3 MB |
| `sum(muladd.(x, w, b))` | same (noise) | 61.0 MB vs 68.7 MB | 1.97 ms vs 2.32 ms | 15.3 MB vs 19.1 MB |
| `2 * sum(x .* w) + sum(b)` | 9.6 ms vs 21.8 ms | 22.9 MB vs 38.1 MB | 1.04 ms vs 1.51 ms | 11.4 MB vs 19.1 MB |
| `sum(x) * sum(w)` | 0.94 ms vs 7.7 ms | 0.8 KB vs 15.3 MB | 0.35 ms vs 0.59 ms | 1.9 KB vs 7.6 MB |

At n = 10^6, broadcasting constants:

| | `Fill` | `PatternVector` with a `FillPattern` |
|:--|:--|:--|
| `a .* b`, `exp.(a)`, `a .* b .+ c` | a `Fill`, O(1) | a `PatternVector`, O(1) |
| `muladd.(a, b, c)`, `muladd.(a, 2, 1)`, `ifelse.(a .> 0, a, b)` | an `Array`: 2.1 ms, 7.6 MB | a `PatternVector`: 20 ns, no allocation |

## Enzyme

A `PatternVector` is an immutable struct: pass it as `Active` (a `Duplicated` shadow stays zero), and pass constant
dense arrays as `Duplicated` or enable `set_runtime_activity`, as for any broadcast in Enzyme.

```@raw html
<div class="fc-hero">
<div class="fc-hero-text">
<p class="fc-eyebrow"><img src="assets/logo.png" alt="" width="28" height="28"> PatternVectors.jl</p>
```

# Structured Vectors, Stored as Patterns

```@raw html
<p class="fc-lede">An immutable AbstractVector that stores a few values and the rule for where they go: O(1) memory, broadcasts evaluated once per distinct value, isbits for GPU kernels, and gradients on the pattern values with Zygote, Mooncake and Enzyme.</p>
<nav class="fc-cta">
<a class="fc-btn fc-btn-primary" href="#Quick-start">Get Started <span aria-hidden="true">→</span></a>
<a class="fc-btn" href="https://github.com/rcalxrc08/PatternVectors.jl"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg> View on GitHub</a>
</nav>
<pre class="fc-hero-code"><code><span class="k">w</span> = PatternVector(n, PaddedEvenOddPattern(1.0, 4.0, 2.0, 1.0))  <span class="c"># Simpson</span>
<span class="k">I</span> = h / 3 * sum(w .* f.(x))      <span class="c"># one fused loop, CPU or GPU</span>
<span class="k">e</span> = exp.(w)                      <span class="c"># exp evaluated 4 times</span></code></pre>
</div>
<figure class="fc-hero-figure">
<img src="assets/figures/hero-patterns-dark.svg" alt="The patterns of PatternVectors.jl, each a vector of 20 elements coloured by value">
<img src="assets/figures/hero-patterns-light.svg" alt="The patterns of PatternVectors.jl, each a vector of 20 elements coloured by value">
<figcaption><span class="fc-legend"><i class="fc-ramp"></i>element value</span>
Read from PatternVectors when this site was built: 20 elements each, and how many values each stores.</figcaption>
</figure>
</div>
```


```@example home
Main.DocumentationFigures.hero_stats()  # hide
```

## What it does

```@raw html
<p class="fc-section-lede">Numerical code is full of long vectors with a few distinct values: quadrature weights, alternating signs, boundary values. <em>A PatternVector keeps the vectorized style and removes the cost.</em></p>
<div class="fc-grid fc-features">
<article class="fc-card"><div class="fc-icon"><svg viewBox="0 0 32 32"><rect x="3" y="12" width="5" height="8" rx="1"/><rect x="9.5" y="12" width="5" height="8" rx="1" class="hl"/><rect x="16" y="12" width="5" height="8" rx="1"/><rect x="22.5" y="12" width="5" height="8" rx="1" class="hl"/></svg></div>
<h3>Seven patterns</h3><p>Zero, fill, initial or final value, padded fill, even–odd, padded even–odd; new ones definable outside the package.</p><p class="fc-why">Simpson weights, signs and boundaries without O(n) memory.</p></article>
<article class="fc-card"><div class="fc-icon"><svg viewBox="0 0 32 32"><rect x="7" y="7" width="18" height="18" rx="3"/><rect x="12" y="12" width="8" height="8" rx="1"/><path d="M11 3v4M16 3v4M21 3v4M11 25v4M16 25v4M21 25v4M3 11h4M3 16h4M3 21h4M25 11h4M25 16h4M25 21h4"/></svg></div>
<h3>isbits</h3><p>A PatternVector stores its length and the pattern values only.</p><p class="fc-why">It goes to a GPU kernel as is, with no specialization of the code.</p></article>
<article class="fc-card"><div class="fc-icon"><svg viewBox="0 0 32 32"><path d="M4 9h24M4 16h24M4 23h24"/><circle cx="10" cy="9" r="2.4"/><circle cx="20" cy="16" r="2.4"/><circle cx="14" cy="23" r="2.4"/></svg></div>
<h3>Broadcasts on patterns</h3><p>Over PatternVectors only, the function is evaluated once per distinct value and the result is a PatternVector.</p><p class="fc-why">exp.(w) costs one call per stored value, whatever the length.</p></article>
<article class="fc-card"><div class="fc-icon"><svg viewBox="0 0 32 32"><rect x="4" y="4" width="7" height="7" rx="1.5"/><rect x="12.5" y="4" width="7" height="7" rx="1.5"/><rect x="21" y="4" width="7" height="7" rx="1.5"/><rect x="4" y="12.5" width="7" height="7" rx="1.5"/><rect x="12.5" y="12.5" width="7" height="7" rx="1.5" class="hl"/><rect x="21" y="12.5" width="7" height="7" rx="1.5"/><path d="M16 21v7"/></svg></div>
<h3>Fused with dense arrays</h3><p>Mixed with an Array or a CuArray, the result has the type of the dense array, computed in one fused pass.</p><p class="fc-why">No temporary arrays, on the CPU and on the GPU.</p></article>
<article class="fc-card"><div class="fc-icon"><svg viewBox="0 0 32 32"><path d="M6 26V14M12 26V8M18 26V17M24 26V11"/><path d="M4 27h24"/></svg></div>
<h3>O(1) sum and ranges</h3><p><code>sum</code> and indexing with ranges return the expected types without iterating.</p><p class="fc-why">Reductions that do not grow with n.</p></article>
<article class="fc-card"><div class="fc-icon"><svg viewBox="0 0 32 32"><path d="M4 26 16 6l12 20z"/><path d="M10 26 16 16l6 10"/></svg></div>
<h3>Differentiable</h3><p>Zygote, Mooncake, Enzyme and ReverseDiff, with gradients accumulated on the pattern values.</p><p class="fc-why">Gradients up to 19 000 times faster than on a dense vector.</p></article>
<article class="fc-card"><div class="fc-icon"><svg viewBox="0 0 32 32"><path d="M16 4v8M16 20v8M4 16h8M20 16h8"/><circle cx="16" cy="16" r="4"/></svg></div>
<h3>Extensible</h3><p>A new pattern is a handful of methods: length, element, ranges, broadcast, sum.</p><p class="fc-why">Your structure, with the same guarantees.</p></article>
<article class="fc-card"><div class="fc-icon"><svg viewBox="0 0 32 32"><path d="M5 17l7 7L27 8"/></svg></div>
<h3>Any number type</h3><p>Float16, Float64, irrationals, dual numbers; GPU arrays.</p><p class="fc-why">No Float64 buffers and no index loops anywhere.</p></article>
</div>
```

## How it fits together

```@raw html
<p class="fc-section-lede">A broadcast on patterns stays a pattern; mixed with a dense array it fuses into one loop; reverse mode accumulates the gradient on the few stored values.</p>
<div class="fc-arch">
<ol class="fc-flow" style="--fc-steps: 6">
<li><span class="fc-node-k">01</span><b>Pattern values</b><span>a few scalars</span></li>
<li class="fc-core"><span class="fc-node-k">02</span><b>PatternVector</b><span><code>PatternVector(n, pattern)</code></span></li>
<li><span class="fc-node-k">03</span><b>Pattern broadcast</b><span>once per distinct value</span></li>
<li><span class="fc-node-k">04</span><b>Mixed broadcast</b><span>with <code>Array</code> or <code>CuArray</code>, fused</span></li>
<li><span class="fc-node-k">05</span><b>Reduction</b><span><code>sum</code>, <code>dot</code>, in O(1) or one pass</span></li>
<li class="fc-out"><span class="fc-node-k">06</span><b>Gradient</b><span>on the pattern values</span></li>
</ol>
</div>
```

## Results

```@raw html
<p class="fc-section-lede">Measured with <code>etc/benchmarks.jl</code> on an Intel i7-8550U, one thread, n = 10⁵; the numbers are in <a href="performance/">Performance</a>.</p>
<div class="fc-tabs">
<input type="radio" name="fc-results" id="fc-results-1" checked><label for="fc-results-1">Gradients</label>
<div class="fc-tab-panels">
<figure class="fc-tab-panel"><img src="assets/figures/gradients-dark.svg" alt="Gradient time with a dense vector and with a PatternVector, per AD backend"><img src="assets/figures/gradients-light.svg" alt="Gradient time with a dense vector and with a PatternVector, per AD backend"><figcaption><b>Gradients.</b> The same weights as a dense vector or as a PatternVector, per backend, log scale. The last case involves only the PatternVector: its cost does not depend on n.</figcaption></figure>
</div>
</div>
```

## Similar packages

- [FillArrays.jl](https://github.com/JuliaArrays/FillArrays.jl): lazy constant arrays (`Fill`, `Zeros`, `Ones`) of any
  dimension, the multidimensional counterpart of `FillPattern` and `ZeroPattern`; broadcasts of functions of three
  arguments (`muladd`, `ifelse`) materialize them, where a `PatternVector` stays one (see [Automatic differentiation](ad.md)).
- [LazyArrays.jl](https://github.com/JuliaArrays/LazyArrays.jl): lazy concatenations and broadcasts; a padded vector
  can be written as `Vcat(x, Fill(a, n - 2), y)`, though broadcasts over it are not reduced to the distinct values and
  alternating layouts are not covered.
- [SparseArrays](https://github.com/JuliaSparse/SparseArrays.jl): vectors that are zero except at a few indices.
- [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl): `isbits` arrays of small, fixed size, for the
  opposite case (few elements, all different).

PatternVectors.jl targets structured vectors of arbitrary length, with broadcasting, GPU and AD support built in.

## Quick start

```@raw html
<p class="fc-section-lede">Five steps, every output computed when this site was built.</p>
```

```@raw html
<div class="fc-step"><span class="fc-step-n">1</span><div><h3>Install the package</h3><p>From LocalRegistry, added once.</p></div></div>
```

```julia
using Pkg
Pkg.Registry.add(Pkg.RegistrySpec(url = "https://github.com/rcalxrc08/LocalRegistry.git"))
Pkg.add("PatternVectors")
```

```@raw html
<div class="fc-step"><span class="fc-step-n">2</span><div><h3>Build two vectors</h3><p>A length and a pattern.</p></div></div>
```

```@example quickstart
using PatternVectors
x = PatternVector(10, EvenOddPattern(0.2, 2.3))
z = PatternVector(10, PaddedEvenOddPattern(0.2, -2.0, 4.0, 2.3))
```

```@raw html
<div class="fc-step"><span class="fc-step-n">3</span><div><h3>Broadcast on patterns</h3><p>sin is evaluated twice, and the result is still a PatternVector.</p></div></div>
```

```@example quickstart
@. sin(x) + exp(z)
```

```@raw html
<div class="fc-step"><span class="fc-step-n">4</span><div><h3>Mix with a dense vector</h3><p>A Vector, in one fused loop.</p></div></div>
```

```@example quickstart
y = randn(10)
@. sin(x) * y + z
```

```@raw html
<div class="fc-step"><span class="fc-step-n">5</span><div><h3>Index and sum</h3><p>Ranges keep the pattern; sum is O(1).</p></div></div>
```

```@example quickstart
z[2:2:8], sum(z)
```

```@raw html
<div class="fc-next">
<a href="usage/"><b>Getting started</b><span>Building, indexing, broadcasting, Simpson's rule.</span></a>
<a href="ad/"><b>Automatic differentiation</b><span>Zygote, Mooncake, Enzyme, ReverseDiff.</span></a>
<a href="performance/"><b>Performance</b><span>Against dense vectors and index loops.</span></a>
<a href="api/"><b>API reference</b><span>Every exported name.</span></a>
</div>
```

## The ecosystem

A `PatternVector` mixes with GPU arrays without any specialization of the code and without scalar indexing:

```julia
using PatternVectors, CUDA
CUDA.allowscalar(false)
x = PatternVector(10, EvenOddPattern(0.2f0, 2.3f0))
y = cu(randn(Float32, 10))
@. sin(x) * y   # CuArray
```

[FinancialFFT.jl](https://github.com/rcalxrc08/FinancialFFT.jl) and
[VibratoMonteCarlo.jl](https://github.com/rcalxrc08/VibratoMonteCarlo.jl) depend on it. More
projects at [rcalxrc08.github.io](https://rcalxrc08.github.io/).

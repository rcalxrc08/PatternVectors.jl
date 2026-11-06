# Performance

## Gradients

Weights `w = [a, b, c, b, c, …, d]` built from four scalars, either as a dense vector (`[a, b, c, d][idx]`) or as
`PatternVector(n, PaddedEvenOddPattern(a, b, c, d))`; gradients with respect to `a, b, c, d` and, when present, the
dense `x`. n = 10^5, Julia 1.13, Intel i7-8550U, single thread; all gradients agree across backends.

| Computation | Storage of `w` | Forward | Zygote | Mooncake | Enzyme |
|:--|:--|--:|--:|--:|--:|
| `sum(w .* x)` | dense | 387 µs | 1.86 ms | 11.6 ms | 1.68 ms |
| | PatternVector | 165 µs | 959 µs | 864 µs | 794 µs |
| `sum(sin.(w .* x) .* exp.(w))` | dense | 2.77 ms | 7.40 ms | 18.8 ms | 9.81 ms |
| | PatternVector | 1.64 ms | 4.69 ms | 12.7 ms | 9.00 ms |
| `sum(exp.(w) .* w)` | dense | 1.27 ms | 4.43 ms | 12.8 ms | 4.80 ms |
| | PatternVector | 67 ns | 2.2 µs | 671 ns | 228 ns |

The last row involves only the `PatternVector`: its cost does not depend on n (the same timings at n = 10^3).

The `PatternVector` is never materialized: `exp.(w)` is evaluated four times instead of n, and the gradient of the
pattern values is accumulated in registers instead of on a dense vector. The script is
[`etc/benchmarks.jl`](https://github.com/rcalxrc08/PatternVectors.jl/blob/main/etc/benchmarks.jl).

A broader suite, covering every pattern with several element types (Float16, Float64, irrationals, dual numbers),
functions and gradients, is in [`benchmark`](https://github.com/rcalxrc08/PatternVectors.jl/tree/main/benchmark) and
runs on CI, which prints its summary tables:

```sh
julia --project=benchmark benchmark/runbenchmarks.jl
```

## Against index loops

The following benchmarks run when this documentation is built, on the machine that builds it.

### Multiplication

```@example perf
using PatternVectors, BenchmarkTools, LinearAlgebra
n = 10_000
x = PatternVector(n, EvenOddPattern(0.2, 2.3))
y = randn(n)
x_c = collect(x)
@btime @. $x * $y;
@btime @. $x_c * $y;
nothing # hide
```

### Alternating signs and sum

```@example perf
function f_std_scalar(f_x)
    sum_z = zero(eltype(f_x))
    for i in eachindex(f_x)
        w = ifelse(isodd(i), 1, -1)
        sum_z += @inbounds w * f_x[i]
    end
    return sum_z
end

function f_std_vec(f_x)
    idx = 1:length(f_x)
    W = @. ifelse(isodd(idx), 1, -1)
    return sum(W .* f_x)
end

function f_std_vec_linear_algebra(f_x)
    idx = 1:length(f_x)
    W = @. ifelse(isodd(idx), 1, -1)
    return dot(W, f_x)
end

function f_apv(f_x)
    W = PatternVector(length(f_x), EvenOddPattern(1, -1))
    return sum(W .* f_x)
end

x = randn(n)
f_x = @. sin(x) + x * cos(x)
@btime f_std_scalar($f_x);
@btime f_std_vec($f_x);
@btime f_std_vec_linear_algebra($f_x);
@btime f_apv($f_x);
nothing # hide
```

### Simpson's rule

```@example perf
n2 = 10_001
function f_simpson_std_scalar(f_x)
    N = length(f_x)
    sum_z = zero(eltype(f_x))
    for i in 1:N
        w = ifelse(i == 1 || i == N, 1 / 3, ifelse(iseven(i), 4 / 3, 2 / 3))
        sum_z += @inbounds w * f_x[i]
    end
    return sum_z
end

function f_simpson_std_vec(f_x)
    N = length(f_x)
    idx = 1:N
    W = @. ifelse((idx == 1) | (idx == N), 1 / 3, ifelse(iseven(idx), 4 / 3, 2 / 3))
    return sum(W .* f_x)
end

function f_simpson_apv(f_x)
    W = PatternVector(length(f_x), PaddedEvenOddPattern(1 / 3, 4 / 3, 2 / 3, 1 / 3))
    return sum(W .* f_x)
end

function f_simpson_apv_linear_algebra(f_x)
    W = PatternVector(length(f_x), PaddedEvenOddPattern(1 / 3, 4 / 3, 2 / 3, 1 / 3))
    return dot(W, f_x)
end

x2 = randn(n2)
f_x2 = @. sin(x2) + x2 * cos(x2)
@btime f_simpson_std_scalar($f_x2);
@btime f_simpson_std_vec($f_x2);
@btime f_simpson_apv($f_x2);
@btime f_simpson_apv_linear_algebra($f_x2);
nothing # hide
```

The `PatternVector` versions match the hand-written loops, which instead do not run on the CUDA.jl stack.

using PatternVectors
using Test, ChainRulesCore, Zygote, CUDA, BenchmarkTools
CUDA.allowscalar(false)
@show "Starting CUDA tests"

@testset "CUDA EvenOddPattern" begin
    N = 11
    av = PatternVector(N, EvenOddPattern(-2.0f0, 3.0f0))
    av_cpu = collect(av)
    av_gpu = cu(av_cpu)
    res_av = @. av * av_gpu
    @test res_av isa CuArray
    res_av_gpu = @. av_gpu * av_gpu
    @test all(collect(res_av) .≈ collect(res_av_gpu))
    res_av = collect(@. muladd(av, av, av_gpu))
    res_av_gpu = collect(@. muladd(av_gpu, av_gpu, av_gpu))
    @test all(@. res_av ≈ res_av_gpu)

    res_av = collect(@. muladd(av, 3.0f0, av_gpu))
    res_av_gpu = collect(@. muladd(av_gpu, 3.0f0, av_gpu))
    @test all(@. res_av ≈ res_av_gpu)
end

@testset "CUDA PaddedEvenOddPattern" begin
    N = 11
    av = PatternVector(N, PaddedEvenOddPattern(-2.0f0, 1.0f0, 50.0f0, 3.0f0))
    av_cpu = collect(av)
    av_gpu = cu(av_cpu)
    res_av = collect(@. av * av_gpu)
    res_av_gpu = collect(@. av_gpu * av_gpu)
    @test all(@. res_av ≈ res_av_gpu)
    res_av = collect(@. muladd(av, av, av_gpu))
    res_av_gpu = collect(@. muladd(av_gpu, av_gpu, av_gpu))
    @test all(@. res_av ≈ res_av_gpu)

    res_av = collect(@. muladd(av, 3.0f0, av_gpu))
    res_av_gpu = collect(@. muladd(av_gpu, 3.0f0, av_gpu))
    @test all(@. res_av ≈ res_av_gpu)
    # mixed precision: Float64 pattern with a Float32 CuArray
    av64 = PatternVector(N, PaddedEvenOddPattern(-2.0, 1.0, 50.0, 3.0))
    @test all(collect(av64 .* av_gpu) .≈ av_cpu .* av_cpu)
end

function my_mul(x, y)
    return x .* y
end

@testset "Zygote EvenOddPattern CUDA" begin
    function f_av(x)
        N = length(x)
        av = ChainRulesCore.@ignore_derivatives PatternVector(N, EvenOddPattern(2.0f0, -8.2f0))
        return sum(my_mul(av, x))
    end
    function f_std(x)
        N = length(x)
        one_minus_one = ChainRulesCore.@ignore_derivatives map(i -> ifelse(isodd(i), 2.0f0, -8.2f0), 1:N)
        av = one_minus_one .* x
        return sum(av)
    end
    function f_std_cu(x)
        N = length(x)
        one_minus_one = ChainRulesCore.@ignore_derivatives cu(map(i -> ifelse(isodd(i), 2.0f0, -8.2f0), 1:N))
        av = one_minus_one .* x
        return sum(av)
    end
    NN = 100_000
    x = ones(Float32, NN)
    res_av = Zygote.gradient(f_av, x)
    res_std = Zygote.gradient(f_std, x)
    @test res_av[1] ≈ res_std[1]
    @show "std"
    @btime $f_av($x)
    @btime $f_std($x)
    @show "grad"
    @btime Zygote.gradient($f_av, $x)
    @btime Zygote.gradient($f_std, $x)

    x_cu = cu(ones(NN))
    res_av = Zygote.gradient(f_av, x_cu)
    res_std = Zygote.gradient(f_std_cu, x_cu)
    # the sum rrule returns a PatternVector cotangent, so the gradient stays a lazy PatternVector
    @test res_av[1] isa PatternVector
    @test collect(res_av[1]) ≈ collect(res_std[1])
    @test collect(res_av[1]) ≈ Zygote.gradient(f_std, x)[1]
    @show "std cuda"
    @btime CUDA.@sync $f_av($x_cu)
    @btime CUDA.@sync $f_std_cu($x_cu)
    @show "grad cuda"
    @btime CUDA.@sync Zygote.gradient($f_av, $x_cu)
    @btime CUDA.@sync Zygote.gradient($f_std_cu, $x_cu)
end

@testset "Zygote PaddedEvenOddPattern CUDA" begin
    function f_avp(x)
        N = length(x)
        av = ChainRulesCore.@ignore_derivatives PatternVector(N, PaddedEvenOddPattern(2.0f0, 1.0f0, 1.0f0, -8.2f0))
        return sum(av .* x)
    end
    NN = 11
    x_cu = cu(ones(NN))
    res_av = Zygote.gradient(f_avp, x_cu)
    @test collect(res_av[1]) ≈ collect(PatternVector(NN, PaddedEvenOddPattern(2.0f0, 1.0f0, 1.0f0, -8.2f0)))
end

# Gradient w.r.t. the pattern values with a dense CuArray: the PatternVector cotangent is a CuArray
@testset "Zygote pattern gradient with CUDA" begin
    f(a, b, c, d, x) = sum(PatternVector(length(x), PaddedEvenOddPattern(a, b, c, d)) .* x)
    NN = 100_000
    x = rand(Float32, NN)
    x_cu = cu(x)
    args = (0.2f0, 2.0f0, -8.2f0, 0.5f0)
    res_cpu = Zygote.gradient(f, args..., x)
    res_gpu = Zygote.gradient(f, args..., x_cu)
    @test all(isapprox.(res_gpu[1:4], res_cpu[1:4]; rtol = 1.0f-4))
    @test collect(res_gpu[5]) ≈ res_cpu[5]
    @show "grad pattern cpu"
    @btime Zygote.gradient($f, $args..., $x)
    @show "grad pattern cuda"
    @btime CUDA.@sync Zygote.gradient($f, $args..., $x_cu)
end

using Mooncake, Statistics
mooncake_gradient(f, args...) = Mooncake.value_and_gradient!!(Mooncake.build_rrule(f, args...), f, args...)

@testset "Mooncake CUDA" begin
    NN = 1000
    x = rand(Float32, NN)
    X = rand(Float32, NN, 3)
    pv(n, a, b, c, d) = PatternVector(n, PaddedEvenOddPattern(a, b, c, d))
    h(u, v) = sin(u) * exp(v) + u^2
    pv_const = pv(NN, 0.2f0, 2.0f0, -8.2f0, 0.5f0)
    p = (0.2f0, 2.0f0, -8.2f0, 0.5f0)
    cases = (
        (x -> sum(pv_const .* x), (x,)),
        ((a, b, c, d, x) -> sum(pv(length(x), a, b, c, d) .* x), (p..., x)),
        ((a, b, c, d, s, x) -> sum(h.(pv(length(x), a, b, c, d), x) .+ s), (p..., 0.3f0, x)),
        ((a, b, c, d, X) -> sum(X .* pv(size(X, 1), a, b, c, d)), (p..., X)),
        ((a, b, c, x) -> sum(PatternVector(length(x), EvenOddPattern(a, b)) .* x .+ PatternVector(length(x), FillPattern(c))), (p[1:3]..., x)),
    )
    for (f, args) in cases
        args_cu = map(a -> a isa Array ? cu(a) : a, args)
        v_cu, g_cu = mooncake_gradient(f, args_cu...)
        v, g = mooncake_gradient(f, args...)
        @test isapprox(v_cu, v; rtol = 1.0f-4)
        for k in 2:length(g)
            @test g_cu[k] isa (args[k-1] isa Array ? CuArray : typeof(g[k]))
            @test isapprox(g_cu[k] isa CuArray ? Array(g_cu[k]) : g_cu[k], g[k]; rtol = 1.0f-4)
        end
    end
    f_mul(a, b, c, d, x) = sum(pv(length(x), a, b, c, d) .* x)
    x_big = rand(Float32, 100_000)
    x_big_cu = cu(x_big)
    rule_cpu = Mooncake.build_rrule(f_mul, p..., x_big)
    rule_gpu = Mooncake.build_rrule(f_mul, p..., x_big_cu)
    @show "Mooncake grad pattern cpu"
    @btime Mooncake.value_and_gradient!!($rule_cpu, $f_mul, $p..., $x_big)
    @show "Mooncake grad pattern cuda"
    @btime CUDA.@sync Mooncake.value_and_gradient!!($rule_gpu, $f_mul, $p..., $x_big_cu)
end

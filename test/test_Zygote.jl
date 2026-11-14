using PatternVectors, Test, ChainRulesCore, Zygote, LinearAlgebra

@testset "Zygote EvenOddPattern" begin
    function f_av(x)
        N = 11
        pattern_iv = EvenOddPattern(x, -8.2 * x)
        av = PatternVector(N, pattern_iv)
        return sum(av)
    end
    function f_std_av(x)
        N = 11
        one_minus_one = ChainRulesCore.@ignore_derivatives @. ifelse(isodd(1:N), 1.0, -8.2)
        av = one_minus_one .* x
        return sum(av)
    end
    x = 3.2
    res_av = Zygote.gradient(f_av, x)
    res_std = Zygote.gradient(f_std_av, x)
    @test res_av[1] ≈ res_std[1]
end

@testset "Zygote FillPattern" begin
    function f_av(x)
        N = 11
        pattern_fill = FillPattern(x)
        av = PatternVector(N, pattern_fill)
        return sum(av)
    end
    function f_std_apv(x)
        N = 11
        one_minus_one = ones(N)
        av = one_minus_one .* x
        return sum(av)
    end
    x = 3.2
    res_av = Zygote.gradient(f_av, x)
    res_std = Zygote.gradient(f_std_apv, x)
    @test res_av[1] ≈ res_std[1]
end

@testset "Zygote ZeroPattern" begin
    function f_av(x)
        N = 11
        pattern_zero = ZeroPattern(x)
        pattern_fill = FillPattern(x)
        av = PatternVector(N, pattern_fill)
        av2 = PatternVector(N, pattern_zero)
        return sum(av .+ av2)
    end
    function f_std_apv(x)
        N = 11
        one_minus_one = ones(N)
        av = one_minus_one .* x
        return sum(av)
    end
    x = 3.2
    res_av = Zygote.gradient(f_av, x)
    res_std = Zygote.gradient(f_std_apv, x)
    @test res_av[1] ≈ res_std[1]
end

@testset "Zygote InitialValuePattern" begin
    function f_av(x)
        N = 11
        pattern_zero = ZeroPattern(x)
        pattern_zero2 = ZeroPattern(0)
        pattern_iv = InitialValuePattern(x, x)
        av = PatternVector(N, pattern_iv)
        av2 = PatternVector(N, pattern_zero)
        av3 = PatternVector(N, pattern_zero2)
        return sum(av .+ av2 .+ av3)
    end
    function f_std_apv(x)
        N = 11
        one_minus_one = ones(N)
        av = one_minus_one .* x
        return sum(av)
    end
    x = 3.2
    res_av = Zygote.gradient(f_av, x)
    res_std = Zygote.gradient(f_std_apv, x)
    @test res_av[1] ≈ res_std[1]
end

@testset "Zygote FinalValuePattern" begin
    function f_av(x)
        N = 11
        pattern_zero = ZeroPattern(x)
        pattern_zero2 = ZeroPattern(0)
        pattern_iv = FinalValuePattern(x, x)
        av = PatternVector(N, pattern_iv)
        av2 = PatternVector(N, pattern_zero)
        av3 = PatternVector(N, pattern_zero2)
        return sum(av .+ av2 .+ av3)
    end
    function f_std_apv(x)
        N = 11
        one_minus_one = ones(N)
        av = one_minus_one .* x
        return sum(av)
    end
    x = 3.2
    res_av = Zygote.gradient(f_av, x)
    res_std = Zygote.gradient(f_std_apv, x)
    @test res_av[1] ≈ res_std[1]
end

@testset "Zygote PaddedFillPattern" begin
    function f_av(x)
        N = 11
        pattern_zero = ZeroPattern(x)
        pattern_zero2 = ZeroPattern(0)
        pattern_iv = PaddedFillPattern(x, x, x)
        av = PatternVector(N, pattern_iv)
        av2 = PatternVector(N, pattern_zero)
        av3 = PatternVector(N, pattern_zero2)
        return sum(av .+ av2 .+ av3)
    end
    function f_std_apv(x)
        N = 11
        one_minus_one = ones(N)
        av = one_minus_one .* x
        return sum(av)
    end
    x = 3.2
    res_av = Zygote.gradient(f_av, x)
    res_std = Zygote.gradient(f_std_apv, x)
    @test res_av[1] ≈ res_std[1]
end

@testset "Zygote PaddedEvenOddPattern" begin
    function f_av(x)
        N = 11
        av = PatternVector(N, PaddedEvenOddPattern(x, -8.2 * x, -5.6 * x, 0.2 * x))
        return sum(av)
    end
    function f_std_apv(x)
        N = 11
        one_minus_one = ChainRulesCore.@ignore_derivatives @. ifelse(isodd(1:N), -5.6, -8.2)
        ChainRulesCore.@ignore_derivatives one_minus_one[1] = 1.0
        ChainRulesCore.@ignore_derivatives one_minus_one[end] = 0.2
        av = one_minus_one .* x
        return sum(av)
    end
    x = 3.2
    res_av = Zygote.gradient(f_av, x)
    res_std = Zygote.gradient(f_std_apv, x)
    @test res_av[1] ≈ res_std[1]
end

@testset "Zygote PaddedEvenOddPattern Deep" begin
    function scalar_f2(x, y, z, k)
        return sin(x + y) * z * exp(k)
    end
    function f_av2(x)
        N = 11
        av = PatternVector(N, PaddedEvenOddPattern(x, -8.2 * x, -5.6 * x, 0.2 * x))
        x1 = ones(N)
        x2 = @. cos(x1)
        x3 = @. cos(x2)
        res = @. scalar_f2(av, x1, x2, x3)
        return sum(res)
    end
    function f_std_apv2(x)
        N = 11
        one_minus_one = ChainRulesCore.@ignore_derivatives @. ifelse(isodd(1:N), -5.6, -8.2)
        ChainRulesCore.@ignore_derivatives one_minus_one[1] = 1.0
        ChainRulesCore.@ignore_derivatives one_minus_one[end] = 0.2
        av = one_minus_one .* x
        x1 = ones(N)
        x2 = @. cos(x1)
        x3 = @. cos(x2)
        res = @. scalar_f2(av, x1, x2, x3)
        return sum(res)
    end
    x = 3.2
    res_av = Zygote.gradient(f_av2, x)
    res_std = Zygote.gradient(f_std_apv2, x)
    @test res_av[1] ≈ res_std[1]
end

@testset "Zygote PaddedEvenOddPattern Broken Example" begin
    function my_mul(x, y)
        return x .* y
    end

    function f_av(a, b, c, d)
        N = 100_000
        x = ones(N)
        av = PatternVector(N, PaddedEvenOddPattern(a, b, c, d))
        return sum(my_mul(av, x))
    end

    function generate_i(i, N, a, b, c, d)
        ifelse(i == 1, a, ifelse(i == N, d, ifelse(iseven(i), b, c)))
    end

    function f_std(a, b, c, d)
        N = 100_000
        x = ones(N)
        one_minus_one = @. generate_i(1:N, N, a, b, c, d)
        av = one_minus_one .* x
        return sum(av)
    end
    a = 0.2
    b = 2.0
    c = -8.2
    d = 0.5
    @show res_av = Zygote.gradient(f_av, a, b, c, d)
    @show res_std = Zygote.gradient(f_std, a, b, c, d)
    res_an = (1.0, 49999.0, 49999.0, 1.0)
    # @test res_av ≈ (1.0, 49999.0, 49999.0, 1.0)
    @test all(res_av .≈ res_an)
    dx = 1e-4
    df_da = (f_std(a + dx, b, c, d) - f_std(a - dx, b, c, d)) / (2 * dx)
    df_db = (f_std(a, b + dx, c, d) - f_std(a, b - dx, c, d)) / (2 * dx)
    df_dc = (f_std(a, b, c + dx, d) - f_std(a, b, c - dx, d)) / (2 * dx)
    df_dd = (f_std(a, b, c, d + dx) - f_std(a, b, c, d - dx)) / (2 * dx)
    @test sum(abs.(res_av .- (df_da, df_db, df_dc, df_dd))) < 1e-2
    @test_broken res_std[1] ≈ res_an[1]
    @test_broken res_std[2] ≈ res_an[2]
    @test_broken res_std[3] ≈ res_an[3]
    @test_broken res_std[4] ≈ res_an[4]
end
# Direct calls of the ChainRules rules: through Zygote their pullbacks get inlined and are not tracked by coverage.
# All patterns are linear in their fields, so d(Δ ⋅ pv)/d(field_j) = Δ ⋅ (pv(field_j + 1) - pv).
@testset "ChainRules pullbacks of patterns" begin
    n = 9
    patterns = (FillPattern(1.5), EvenOddPattern(1.5, -2.0), InitialValuePattern(1.5, -2.0), FinalValuePattern(1.5, -2.0),
        PaddedFillPattern(1.5, -2.0, 0.3), PaddedEvenOddPattern(1.5, -2.0, 0.3, 4.0))
    Δ = rand(n)
    for pattern in patterns, Δy in (Δ, PatternVector(n, PaddedEvenOddPattern(0.7, -1.1, 2.3, 0.4)))
        P = typeof(pattern)
        constructor = Base.typename(P).wrapper
        fields = ntuple(j -> getfield(pattern, j), fieldcount(P))
        bumped(j) = constructor(ntuple(k -> fields[k] + (k == j), fieldcount(P))...)
        expected = ntuple(j -> sum(collect(Δy) .* (collect(PatternVector(n, bumped(j))) .- collect(PatternVector(n, pattern)))), fieldcount(P))
        y, pv_pb = ChainRulesCore.rrule(PatternVector, n, pattern)
        @test y == PatternVector(n, pattern)
        dpattern = pv_pb(Δy)[3]
        @test all(ntuple(j -> getfield(dpattern, j), fieldcount(P)) .≈ expected)
        p, p_pb = ChainRulesCore.rrule(constructor, fields...)
        @test p == pattern
        @test all(p_pb(dpattern)[2:end] .≈ expected)
    end
    y, z_pb = ChainRulesCore.rrule(ZeroPattern, 1.0)
    @test y == ZeroPattern(1.0)
    @test z_pb(nothing) == (NoTangent(), NoTangent())
    @test PatternVectors.pattern_to_vector_pullback(ZeroPattern{Float64}, Δ, n) == ZeroPattern(0.0)
end

@testset "Keyword sum of PatternVector" begin
    for pattern in (ZeroPattern(1.0), FillPattern(1.5), EvenOddPattern(1.5, -2.0), PaddedEvenOddPattern(1.5, -2.0, 0.3, 4.0))
        pv = PatternVector(9, pattern)
        @test sum(pv) ≈ sum(collect(pv))
        @test sum(pv; dims = :) ≈ sum(collect(pv))
        @test sum(pv; dims = 1) ≈ sum(collect(pv); dims = 1)
    end
    # Zygote's broadcast pullback reduces scalar arguments with `sum(Δ; dims = :)`: it must not iterate over n
    function f_chain(a, n)
        p = PatternVector(n, PaddedEvenOddPattern(a, 2a, 3a, 4a))
        q = PatternVector(n, EvenOddPattern(sin(a), cos(a)))
        return sum(@. exp(p) * q - a / 3)
    end
    a = 0.2
    n_small = 11
    function f_chain_dense(a, n)
        p = collect(PatternVector(n, PaddedEvenOddPattern(a, 2a, 3a, 4a)))
        q = collect(PatternVector(n, EvenOddPattern(sin(a), cos(a))))
        return sum(@. exp(p) * q - a / 3)
    end
    @test Zygote.gradient(f_chain, a, n_small)[1] ≈ Zygote.gradient(f_chain_dense, a, n_small)[1]
    @test isfinite(Zygote.gradient(f_chain, a, 10^12)[1])
end

# Zygote ≥ 0.7.17 passes the cotangent of a scalar as a deferred Thunk (here, of `sum(x) .* 2`): the rules unthunk it,
# so that the cotangent of `sum` stays a PatternVector of numbers (isbits, usable in a GPU kernel) and not a host Vector
# made by Zygote's `map` over a PatternVector of thunks (#19).
@testset "Zygote deferred cotangents" begin
    x = collect(1.0:5.0)
    for f in (x -> sum(x) .* 2.0, x -> (sum(x) .* 2.0) .* 3.0, x -> sum(x) .* sum(x), x -> max.(sum(x) .- 3.0, 0.0) .* 2.0)
        g = Zygote.gradient(f, x)[1]
        @test g isa PatternVector{Float64, FillPattern{Float64}}
        @test isbits(g)
        @test g ≈ Zygote.gradient(y -> f(collect(y)), x)[1]
    end
    # the pattern constructors receive deferred cotangents too: each pattern is linear in a, with these coefficients
    for (n, P, c) in ((5, FillPattern, (1.0,)), (6, EvenOddPattern, (1.0, 2.0)), (6, PaddedEvenOddPattern, (1.0, 2.0, 3.0, 4.0)), (6, PaddedFillPattern, (1.0, 2.0, 3.0)), (6, InitialValuePattern, (1.0, 2.0)), (6, FinalValuePattern, (1.0, 2.0)))
        f(a) = sum(PatternVector(n, P((c .* a)...))) .* 2.0
        @test Zygote.gradient(f, 0.3)[1] ≈ 2 * sum(collect(PatternVector(n, P(c...))))
    end
end

# dot: the weighted sums of quadratures (#21). Same gradients as sum(conj(w) .* y); the cotangent of the dense argument
# is a PatternVector; constant integer weights get no cotangent.
@testset "Zygote dot" begin
    n = 11
    y = collect(range(0.1, 1.1, length = n))
    simpson = PatternVector(n, PaddedEvenOddPattern(Int8(1), Int8(4), Int8(2), Int8(1)))
    g_dot = Zygote.gradient(a -> dot(simpson, exp.(a .* y)), 0.7)[1]
    @test g_dot ≈ Zygote.gradient(a -> sum(simpson .* exp.(a .* y)), 0.7)[1]
    @test Zygote.gradient(a -> dot(exp.(a .* y), simpson), 0.7)[1] ≈ g_dot
    ȳ = Zygote.gradient(z -> dot(simpson, z), y)[1]
    @test ȳ isa PatternVector && isbits(ȳ) && ȳ == collect(simpson)
    @test Zygote.gradient(z -> dot(simpson, z) .* 2.0, y)[1] isa PatternVector  # deferred cotangent
    # differentiable patterns, both orders and two PatternVectors
    f_pv(a, b) = dot(PatternVector(n, EvenOddPattern(a, b)), y)
    f_dense(a, b) = sum(PatternVector(n, EvenOddPattern(a, b)) .* y)
    @test all(Zygote.gradient(f_pv, 0.3, 0.5) .≈ Zygote.gradient(f_dense, 0.3, 0.5))
    @test all(Zygote.gradient((a, b) -> dot(y, PatternVector(n, EvenOddPattern(a, b))), 0.3, 0.5) .≈ Zygote.gradient(f_dense, 0.3, 0.5))
    f_pp(a, b) = dot(PatternVector(n, PaddedFillPattern(a, b, a)), PatternVector(n, EvenOddPattern(b, a)))
    f_pp_dense(a, b) = sum(collect(PatternVector(n, PaddedFillPattern(a, b, a))) .* collect(PatternVector(n, EvenOddPattern(b, a))))
    @test all(Zygote.gradient(f_pp, 0.3, 0.5) .≈ Zygote.gradient(f_pp_dense, 0.3, 0.5))
    # complex: dot conjugates its first argument
    yc = y .+ im .* reverse(y)
    @test Zygote.gradient(a -> real(dot(PatternVector(n, FillPattern(a)), yc)), 0.3)[1] ≈ Zygote.gradient(a -> real(sum(conj.(PatternVector(n, FillPattern(a))) .* yc)), 0.3)[1]
    @test Zygote.gradient(z -> real(dot(simpson, z)), yc)[1] ≈ Zygote.gradient(z -> real(sum(simpson .* z)), yc)[1]
    # the rule called directly (Zygote inlines it, and coverage would not count it)
    val, pb = ChainRulesCore.rrule(dot, simpson, y)
    @test val ≈ dot(collect(simpson), y)
    _, w̄, ȳ_direct = pb(2.0)
    @test w̄ isa NoTangent && unthunk(ȳ_direct) == 2 .* collect(simpson)
    _, ȳ2, w̄2 = ChainRulesCore.rrule(dot, y, simpson)[2](ChainRulesCore.@thunk(2.0))
    @test w̄2 isa NoTangent && unthunk(ȳ2) == 2 .* collect(simpson)
end

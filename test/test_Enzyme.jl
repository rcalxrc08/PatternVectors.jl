using PatternVectors, Test, Enzyme, Mooncake, Zygote

# Enzyme needs no extension: it differentiates PatternVector structurally. A PatternVector is an immutable struct,
# so it is passed as Active and its gradient is a PatternVector whose pattern holds the gradient of the pattern
# fields (Enzyme zeroes the integer field, so the gradient has length 0: read it through `.pattern`).
enzyme_pattern_fields(g::PatternVector) = Tuple(getfield(g.pattern, k) for k in fieldnames(typeof(g.pattern)))
mooncake_pattern_fields(t::Mooncake.Tangent) = values(t.fields.pattern.fields)
mooncake_pattern_fields(::Mooncake.NoTangent) = ()

@testset "Enzyme vs Zygote scalar through $(nameof(typeof(make(1.0))))" for make in (
        x -> FillPattern(x),
        x -> EvenOddPattern(x, -8.2 * x),
        x -> InitialValuePattern(x, 2 * x),
        x -> FinalValuePattern(x, 2 * x),
        x -> PaddedFillPattern(x, 2 * x, -x),
        x -> PaddedEvenOddPattern(x, -8.2 * x, -5.6 * x, 0.2 * x),
    )
    n = 11
    dense = collect(1.0:n)
    for f in (
            (x, _) -> sum(PatternVector(n, make(x))),
            (x, _) -> sum(PatternVector(n, make(x)) .+ PatternVector(n, ZeroPattern(x))),
            (x, d) -> sum(sin.(PatternVector(n, make(x))) .* d),
            (x, d) -> (p = PatternVector(n, make(x)); sum(@. cos(p * d) * exp(d))),
        )
        dd = zero(dense)
        dx = Enzyme.autodiff(Reverse, Const(f), Active, Active(3.2), Duplicated(dense, dd))[1][1]
        zx, zd = Zygote.gradient(f, 3.2, dense)
        @test dx ≈ zx
        @test dd ≈ something(zd, zero(dense))
    end
    # A constant dense array in a broadcast needs runtime activity, exactly as with plain Arrays
    f_const(x, d) = sum(sin.(PatternVector(n, make(x))) .* d)
    @test Enzyme.autodiff(set_runtime_activity(Reverse), Const(f_const), Active, Active(3.2), Const(dense))[1][1] ≈ Zygote.gradient(f_const, 3.2, dense)[1]
end

@testset "Enzyme PatternVector gradient vs Mooncake $(nameof(typeof(pattern)))" for pattern in (
        FillPattern(1.5),
        EvenOddPattern(1.0, 2.0),
        InitialValuePattern(1.0, 2.0),
        FinalValuePattern(1.0, 2.0),
        PaddedFillPattern(1.0, 2.0, 3.0),
        PaddedEvenOddPattern(1.0, 2.0, 3.0, 4.0),
        ZeroPattern(0.0),
    )
    n = 11
    h(u, v) = sin(u) * exp(v) + u * v
    pv = PatternVector(n, pattern)
    x = rand(n)
    X = rand(n, 3)
    for f in (
            (p, x, X) -> sum(p),
            (p, x, X) -> sum(p .* x),
            (p, x, X) -> sum(h.(p, x)),
            (p, x, X) -> sum(X .* p),
            (p, x, X) -> sum(p .* p .+ 2),
            (p, x, X) -> sum(cos.(p .* x) .+ p),
        )
        dx = zero(x)
        dX = zero(X)
        dpv = Enzyme.autodiff(Reverse, Const(f), Active, Active(pv), Duplicated(x, dx), Duplicated(X, dX))[1][1]
        rule = Mooncake.build_rrule(f, pv, x, X)
        _, (_, mpv, mx, mX) = Mooncake.value_and_gradient!!(rule, f, pv, x, X)
        @test all(enzyme_pattern_fields(dpv) .≈ mooncake_pattern_fields(mpv))
        @test dx ≈ mx
        @test dX ≈ mX
    end
end

@testset "Enzyme PatternVector gradient is size independent" begin
    n = 10^12
    pv = PatternVector(n, PaddedEvenOddPattern(1.0, 2.0, 3.0, 4.0))
    @test enzyme_pattern_fields(Enzyme.autodiff(Reverse, sum, Active, Active(pv))[1][1]) == (1.0, (n - 2) / 2, (n - 2) / 2, 1.0)
    f(p) = sum(p .* p)
    @test enzyme_pattern_fields(Enzyme.autodiff(Reverse, f, Active, Active(pv))[1][1]) == (2.0, 4.0 * (n - 2) / 2, 6.0 * (n - 2) / 2, 8.0)
end

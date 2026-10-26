using PatternVectors, Test, Mooncake, Zygote

function mooncake_gradient(f, x...)
    rule = Mooncake.build_rrule(f, x...)
    _, grad = Mooncake.value_and_gradient!!(rule, f, x...)
    return grad[2:end]
end

@testset "Mooncake tangent of PatternVector is structural" begin
    for pattern in (FillPattern(1.0), EvenOddPattern(1.0, 2.0), PaddedEvenOddPattern(1.0, 2.0, 3.0, 4.0))
        T_pv = Mooncake.tangent_type(typeof(PatternVector(11, pattern)))
        T_pattern = Mooncake.tangent_type(typeof(pattern))
        @test T_pv == Mooncake.Tangent{@NamedTuple{n::Mooncake.NoTangent, pattern::T_pattern}}
    end
end

@testset "Mooncake PatternVector cotangent" begin
    av = PatternVector(11, PaddedEvenOddPattern(1.0, 2.0, 3.0, 4.0))
    f(v) = sum(v .* ones(11))
    rule = Mooncake.build_rrule(f, av)
    _, (_, dav) = Mooncake.value_and_gradient!!(rule, f, av)
    dp = dav.fields.pattern.fields
    @test dp.bound_initial_value ≈ 1.0
    @test dp.value_even ≈ 5.0
    @test dp.value_odd ≈ 4.0
    @test dp.bound_final_value ≈ 1.0
end

N = 11
f_eo(x) = sum(PatternVector(N, EvenOddPattern(x, -8.2 * x)))
f_fill(x) = sum(PatternVector(N, FillPattern(x)))
f_zero(x) = sum(PatternVector(N, FillPattern(x)) .+ PatternVector(N, ZeroPattern(x)))
f_iv(x) = sum(PatternVector(N, InitialValuePattern(x, x)) .+ PatternVector(N, ZeroPattern(x)) .+ PatternVector(N, ZeroPattern(0)))
f_fv(x) = sum(PatternVector(N, FinalValuePattern(x, x)) .+ PatternVector(N, ZeroPattern(x)) .+ PatternVector(N, ZeroPattern(0)))
f_pf(x) = sum(PatternVector(N, PaddedFillPattern(x, x, x)) .+ PatternVector(N, ZeroPattern(x)) .+ PatternVector(N, ZeroPattern(0)))
f_peo(x) = sum(PatternVector(N, PaddedEvenOddPattern(x, -8.2 * x, -5.6 * x, 0.2 * x)))
scalar_f2(x, y, z, k) = sin(x + y) * z * exp(k)
function f_peo_deep(x)
    av = PatternVector(N, PaddedEvenOddPattern(x, -8.2 * x, -5.6 * x, 0.2 * x))
    x1 = ones(N)
    x2 = @. cos(x1)
    x3 = @. cos(x2)
    return sum(@. scalar_f2(av, x1, x2, x3))
end

@testset "Mooncake vs Zygote $name" for (name, f) in (
        "EvenOddPattern" => f_eo,
        "FillPattern" => f_fill,
        "ZeroPattern" => f_zero,
        "InitialValuePattern" => f_iv,
        "FinalValuePattern" => f_fv,
        "PaddedFillPattern" => f_pf,
        "PaddedEvenOddPattern" => f_peo,
        "PaddedEvenOddPattern Deep" => f_peo_deep,
    )
    x = 3.2
    @test mooncake_gradient(f, x)[1] ≈ Zygote.gradient(f, x)[1]
end

@testset "Mooncake PaddedEvenOddPattern multiple inputs" begin
    my_mul(x, y) = x .* y
    f_av(a, b, c, d) = sum(my_mul(PatternVector(10, PaddedEvenOddPattern(a, b, c, d)), ones(10)))
    res = mooncake_gradient(f_av, 0.2, 2.0, -8.2, 0.5)
    @test all(res .≈ (1.0, 4.0, 4.0, 1.0))
end

@testset "Mooncake mixed broadcast rule" begin
    mm = PatternVectors.materialize_mixture
    DS = Base.Broadcast.DefaultArrayStyle{1}
    pvT = typeof(PatternVector(10, PaddedEvenOddPattern(1.0, 2.0, 3.0, 4.0)))
    is_prim(sig) = Mooncake.is_primitive(Mooncake.DefaultCtx, Mooncake.ReverseMode, sig, Base.get_world_counter())
    mulstrict(u::Float64, v::Float64) = u * v
    @test is_prim(Tuple{typeof(mm), DS, typeof(*), pvT, Vector{Float64}})
    @test is_prim(Tuple{typeof(mm), DS, typeof(Base.literal_pow), Base.RefValue{typeof(^)}, pvT, Base.RefValue{Val{2}}})
    @test !is_prim(Tuple{typeof(mm), DS, typeof(mulstrict), pvT, Vector{Float64}}) # NDual not accepted: derived rule
    @test !is_prim(Tuple{typeof(mm), DS, typeof(*), pvT, Vector{Float32}})         # mixed precision: derived rule
    @test !is_prim(Tuple{typeof(mm), DS, typeof(*), pvT, Base.RefValue{Float64}})   # differentiable Ref: derived rule

    gen(i, n, a, b, c, d) = ifelse(i == 1, a, ifelse(i == n, d, ifelse(isodd(i), c, b)))
    h(u, v, w) = sin(u + v) * w + u^2 / 3
    pv(n, a, b, c, d) = PatternVector(n, PaddedEvenOddPattern(a, b, c, d))
    dv(n, a, b, c, d) = gen.(1:n, n, a, b, c, d)
    cases = (
        ((a, b, c, d, s, x, X) -> sum(pv(length(x), a, b, c, d) .* x), (a, b, c, d, s, x, X) -> sum(dv(length(x), a, b, c, d) .* x)),
        ((a, b, c, d, s, x, X) -> sum(mulstrict.(pv(length(x), a, b, c, d), x)), (a, b, c, d, s, x, X) -> sum(dv(length(x), a, b, c, d) .* x)),
        ((a, b, c, d, s, x, X) -> sum(h.(pv(length(x), a, b, c, d), x, s)), (a, b, c, d, s, x, X) -> sum(h.(dv(length(x), a, b, c, d), x, s))),
        ((a, b, c, d, s, x, X) -> (p1 = PatternVector(length(x), EvenOddPattern(a, b)); p2 = PatternVector(length(x), FillPattern(d)); sum(@. exp(p1 * c) * sin(x)^2 + x * p2)),
         (a, b, c, d, s, x, X) -> (r = 1:length(x); sum(@. exp(ifelse(isodd(r), a, b) * c) * sin(x)^2 + x * d))),
        ((a, b, c, d, s, x, X) -> sum(X .* pv(size(X, 1), a, b, c, d) .+ s), (a, b, c, d, s, x, X) -> sum(X .* dv(size(X, 1), a, b, c, d) .+ s)),
        ((a, b, c, d, s, x, X) -> sum(X[1:1, :] .* pv(size(X, 1), a, b, c, d)), (a, b, c, d, s, x, X) -> sum(X[1:1, :] .* dv(size(X, 1), a, b, c, d))),
    )
    args = (0.3, -1.2, 0.7, 2.1, 0.4, rand(9), rand(9, 4))
    strategy = Base.get_extension(PatternVectors, :MooncakeExt).PARTIALS_STRATEGY
    default_strategy = strategy[]
    for partials_strategy in (:store, :recompute)
        strategy[] = partials_strategy
        for (f_pv, f_ref) in cases
            @test all(mooncake_gradient(f_pv, args...) .≈ mooncake_gradient(f_ref, args...))
        end
    end
    strategy[] = default_strategy
end

# The broadcast-based reverse pass serves non-Array arrays (e.g. CuArray). A broadcast style other than
# DefaultArrayStyle over plain Arrays selects it on CPU: compare it with the fused loop through rrule!! itself.
struct GenericArrayStyle{N} <: Base.Broadcast.AbstractArrayStyle{N} end
Base.similar(bc::Base.Broadcast.Broadcasted{GenericArrayStyle{N}}, ::Type{T}) where {N, T} = similar(Array{T}, axes(bc))

@testset "Mooncake broadcast-based reverse pass" begin
    numbers(r::Mooncake.RData) = numbers(r.data)
    numbers(r::Union{Tuple, NamedTuple}) = reduce(vcat, map(numbers, collect(r)); init = Float64[])
    numbers(r::Number) = [r]
    numbers(::Mooncake.NoRData) = Float64[]
    function run_rule(style, f, xs...)
        args = map(Mooncake.zero_fcodual, xs)
        y, pb = Mooncake.rrule!!(Mooncake.zero_fcodual(PatternVectors.materialize_mixture), Mooncake.zero_fcodual(style), Mooncake.zero_fcodual(f), args...)
        dy = Mooncake.tangent(y)
        dy .= reshape(range(0.5, 1.5; length = length(dy)), size(dy))
        return copy(Mooncake.primal(y)), numbers(pb(Mooncake.NoRData())), map(Mooncake.tangent, args)
    end
    n = 9
    pv = PatternVector(n, PaddedEvenOddPattern(0.3, -1.2, 0.7, 2.1))
    cases = (
        (*, (pv, rand(n))),
        ((u, v, w) -> sin(u) * v + w, (pv, rand(n), 0.4)),
        (*, (rand(n, 3), pv)),
        (*, (rand(1, 3), pv)),
        (Base.literal_pow, (Ref(^), pv, Ref(Val(2)))),
    )
    for (f, xs) in cases
        N = maximum(x -> x isa AbstractArray ? ndims(x) : 0, xs)
        y_broadcast, r_broadcast, d_broadcast = run_rule(GenericArrayStyle{N}(), f, xs...)
        y_fused, r_fused, d_fused = run_rule(Base.Broadcast.DefaultArrayStyle{N}(), f, xs...)
        @test y_broadcast ≈ y_fused
        @test r_broadcast ≈ r_fused
        for (a, b) in zip(d_broadcast, d_fused)
            a isa Array && @test a ≈ b
        end
    end
end

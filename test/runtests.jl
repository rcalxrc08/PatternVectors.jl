using PatternVectors
using Test, Dates, ChainRulesCore, Zygote, LazyArrays, SparseArrays

struct ZeroDimensionalVector{T} <: AbstractArray{T, 0}
    value::T
    function ZeroDimensionalVector(value::T) where {T}
        return new{T}(value)
    end
end

### Implementation of the array interface
Base.size(::ZeroDimensionalVector) = ()

function Base.getindex(x::ZeroDimensionalVector, ind::Int)
    throw(BoundsError(x, ind))
end
function Base.getindex(A::ZeroDimensionalVector)
    return A.value
end

# IO
Base.showarg(io::IO, A::ZeroDimensionalVector, _) = print(io, typeof(A))

# Broacasting relation against other arrays
struct ArrayStyleZeroDimVector <: Broadcast.AbstractArrayStyle{0} end
Base.BroadcastStyle(::Type{<:ZeroDimensionalVector{T}}) where {T} = ArrayStyleZeroDimVector()
Base.BroadcastStyle(::ArrayStyleZeroDimVector, b::Base.Broadcast.Style{Tuple}) = b
function Base.BroadcastStyle(::ArrayStyleZeroDimVector, b::Broadcast.AbstractArrayStyle{N}) where {N}
    return b
end
function Base.BroadcastStyle(::ArrayStyleZeroDimVector, b::Broadcast.DefaultArrayStyle{N}) where {N}
    return b
end

const AlternateVector_test{T} = PatternVector{T, PatternVectors.EvenOddPattern{T}};
const AlternatePaddedVector_test{T} = PatternVector{T, PatternVectors.PaddedEvenOddPattern{T}};

@testset "EvenOddPattern" begin
    @test_throws "length of PatternVector for pattern" PatternVector(1, EvenOddPattern(0, 0))
    N = 11

    av = PatternVector(N, EvenOddPattern(-2.0, 3.0))
    @test PatternVectors.pattern_minimum_size(PatternVectors.EvenOddPattern(0, 0)) == 2
    @test av[1] == -2.0
    @test av[2] == 3.0
    @test av[end] == -2.0
    Base.showarg(Core.CoreSTDOUT(), av, nothing)
    println(av)
    @show av
    av_c = collect(av)
    @test typeof(av[1:2]) <: AlternateVector_test
    @test typeof(av[:]) <: AlternateVector_test
    @test typeof(1 .+ av) <: AlternateVector_test
    @test typeof(av .+ 1) <: AlternateVector_test
    @test typeof(sin.(av)) <: AlternateVector_test
    @test !(typeof(av .+ tuple(ones(N)...)) <: AlternateVector_test)
    @test !(typeof(tuple(ones(N)...) .+ av) <: AlternateVector_test)
    @test typeof(@. sin(av) * av + 1 + av) <: AlternateVector_test
    @test typeof(@. sin(av) * av + 1 + exp(av)) <: AlternateVector_test
    @test typeof(@. sin(av) * av * av + 1 + exp(av)) <: AlternateVector_test
    @test typeof(@. sin(cos(av)) * av * av + exp(1) + exp(av)) <: AlternateVector_test
    @test typeof(@. 2 + sin(av) * av + 1 + exp(av)) <: AlternateVector_test
    @test all(@. av ≈ av_c)
    @test all(@. sin(av) ≈ sin(av_c))
    @test all(@. exp(av) + av ≈ exp(av_c) + av_c)
    @test all(@. exp(av) + av_c ≈ exp(av_c) + av)
    @test all(@. exp(av) + av_c ≈ exp(av_c) + av_c)
    @test all(@. exp(av) + av_c + av * av_c ≈ exp(av_c) + av_c + av * av_c)
    @test all(@. exp(av) + av + av * av ≈ exp(av_c) + av_c + av * av_c)
    @test sum(av) ≈ sum(av_c)
    av_d = PatternVector(N, EvenOddPattern(Dates.Date(1992, 1, 1), Dates.Date(1992, 10, 1)))
    av_d_1 = collect(av_d)
    one_day = Dates.Day(1)
    res_av_with_ref = @. av_d + one_day
    @test all(@. res_av_with_ref == av_d_1 + one_day)
    #test sparse
    sparse_p = spzeros(Float64, N)
    sparse_p[1] = 4.0
    @test typeof(sin.(av .* sparse_p)) <: typeof(sparse_p)
    #test lazy arrays
    D = LazyArray(@~ exp.(1:N))
    D_c = collect(D)
    res_mul_lazy_1 = D .* av
    res_mul_lazy_2 = av .* D
    res_mul = av_c .* D_c
    @test all(@. res_mul_lazy_1 ≈ res_mul)
    @test all(@. res_mul_lazy_2 ≈ res_mul)

    x_zero_dim = ZeroDimensionalVector(1)
    res_lazy_1_zero_dim = x_zero_dim .* av
    res_lazy_2_zero_dim = av .* x_zero_dim
    @test all(@. res_lazy_1_zero_dim ≈ res_lazy_2_zero_dim)

    x_zero_dim_r = Array{Float64, 0}(undef)
    x_zero_dim_r .= 0
    res_zero_d_r = @. x_zero_dim_r * av
    res_zero_d = @. x_zero_dim_r * av_c
    @test all(@. res_zero_d ≈ res_zero_d_r)

    av_int = PatternVector(N, EvenOddPattern(-2, 3))
    @test all(@. exp(av) + av_int ≈ exp(av_c) + av_int)
    @test all(@. exp(av) + av_int ≈ av_int + exp(av))
    @test all(@. av + av_int ≈ av_int + av)
end

@testset "FillPattern" begin
    @test_throws "length of PatternVector for pattern" PatternVector(0, FillPattern(0))
    N = 11
    val = 10.0
    av = PatternVector(N, FillPattern(val))
    @test PatternVectors.pattern_minimum_size(FillPattern(0)) == 1
    @test av[1] == val
    @test av[2] == val
    @test av[end] == val
    Base.showarg(Core.CoreSTDOUT(), av, nothing)
    println(av)
    @show av
    av_c = collect(av)
    @test sum(av) ≈ sum(av_c)
    @test typeof(av[1:2]) <: PatternVector
    @test typeof(av[:]) <: PatternVector
    @test typeof(1 .+ av) <: PatternVector
    @test typeof(av .+ 1) <: PatternVector
    @test typeof(sin.(av)) <: PatternVector
    @test !(typeof(av .+ tuple(ones(N)...)) <: PatternVector)
    @test !(typeof(tuple(ones(N)...) .+ av) <: PatternVector)
    @test typeof(@. sin(av) * av + 1 + av) <: PatternVector
    @test typeof(@. sin(av) * av + 1 + exp(av)) <: PatternVector
    @test typeof(@. sin(av) * av * av + 1 + exp(av)) <: PatternVector
    @test typeof(@. sin(cos(av)) * av * av + exp(1) + exp(av)) <: PatternVector
    @test typeof(@. 2 + sin(av) * av + 1 + exp(av)) <: PatternVector
    @test all(@. av ≈ av_c)
    @test all(@. sin(av) ≈ sin(av_c))
    @test all(@. exp(av) + av ≈ exp(av_c) + av_c)
    @test all(@. exp(av) + av_c ≈ exp(av_c) + av)
    @test all(@. exp(av) + av_c ≈ exp(av_c) + av_c)
    @test all(@. exp(av) + av_c + av * av_c ≈ exp(av_c) + av_c + av * av_c)
    @test all(@. exp(av) + av + av * av ≈ exp(av_c) + av_c + av * av_c)
    @test typeof((PatternVector(N, FillPattern(0.0f0)) .+ av).pattern) <: FillPattern
end
@testset "ZeroPattern" begin
    @test_throws "length of PatternVector for pattern" PatternVector(0, ZeroPattern(0))
    N = 11
    val = 0.0
    av = PatternVector(N, PatternVectors.ZeroPattern(0.0))
    @test PatternVectors.pattern_minimum_size(ZeroPattern(0)) == 1
    @test av[1] == val
    @test av[2] == val
    @test av[end] == val
    Base.showarg(Core.CoreSTDOUT(), av, nothing)
    println(av)
    @show av
    av_c = collect(av)
    @test sum(av) ≈ sum(av_c)
    @test typeof(av[1:2]) <: PatternVector
    @test typeof(av[:]) <: PatternVector
    @test typeof(1 .+ av) <: PatternVector
    @test typeof(av .+ 1) <: PatternVector
    @test typeof(sin.(av)) <: PatternVector
    @test !(typeof(av .+ tuple(ones(N)...)) <: PatternVector)
    @test !(typeof(tuple(ones(N)...) .+ av) <: PatternVector)
    @test typeof(@. sin(av) * av + 1 + av) <: PatternVector
    @test typeof(@. sin(av) * av + 1 + exp(av)) <: PatternVector
    @test typeof(@. sin(av) * av * av + 1 + exp(av)) <: PatternVector
    @test typeof(@. sin(cos(av)) * av * av + exp(1) + exp(av)) <: PatternVector
    @test typeof(@. 2 + sin(av) * av + 1 + exp(av)) <: PatternVector
    @test all(@. av ≈ av_c)
    @test all(@. sin(av) ≈ sin(av_c))
    @test all(@. exp(av) + av ≈ exp(av_c) + av_c)
    @test all(@. exp(av) + av_c ≈ exp(av_c) + av)
    @test all(@. exp(av) + av_c ≈ exp(av_c) + av_c)
    @test all(@. exp(av) + av_c + av * av_c ≈ exp(av_c) + av_c + av * av_c)
    @test all(@. exp(av) + av + av * av ≈ exp(av_c) + av_c + av * av_c)
    av2 = PatternVector(N, FillPattern(val))
    @test all(@. exp(av) + av + av2 ≈ exp(av_c) + av_c + av2)
    @test typeof((PatternVector(N, ZeroPattern(0.0f0)) .+ av).pattern) <: ZeroPattern
end
@testset "InitialValuePattern" begin
    @test_throws "length of PatternVector for pattern" PatternVector(0, InitialValuePattern(0, 0))
    N = 11
    val = 0.0
    av = PatternVector(N, PatternVectors.InitialValuePattern(val, val + 1))
    av33 = PatternVector(N, PatternVectors.InitialValuePattern(0, 1))
    @test PatternVectors.pattern_minimum_size(ZeroPattern(0)) == 1
    @test av[1] == val
    @test av[2] == val + 1
    @test av[end] == val + 1
    Base.showarg(Core.CoreSTDOUT(), av, nothing)
    println(av)
    @show av
    av_c = collect(av)
    @test sum(av) ≈ sum(av_c)
    @test typeof(av[1:2]) <: PatternVector
    @test typeof(av[:]) <: PatternVector
    @test typeof(1 .+ av) <: PatternVector
    @test typeof(av .+ 1) <: PatternVector
    @test typeof(sin.(av)) <: PatternVector
    @test !(typeof(av .+ tuple(ones(N)...)) <: PatternVector)
    @test !(typeof(tuple(ones(N)...) .+ av) <: PatternVector)
    @test typeof(@. sin(av) * av + 1 + av) <: PatternVector
    @test typeof(@. sin(av) * av + 1 + exp(av)) <: PatternVector
    @test typeof(@. sin(av) * av * av + 1 + exp(av)) <: PatternVector
    @test typeof(@. sin(cos(av)) * av * av + exp(1) + exp(av)) <: PatternVector
    @test typeof(@. 2 + sin(av) * av + 1 + exp(av)) <: PatternVector
    @test all(@. av ≈ av_c)
    @test all(@. sin(av) ≈ sin(av_c))
    @test all(@. exp(av) + av ≈ exp(av_c) + av_c)
    @test all(@. exp(av) + av_c ≈ exp(av_c) + av)
    @test all(@. exp(av) + av_c ≈ exp(av_c) + av_c)
    @test all(@. exp(av) + av_c + av * av_c ≈ exp(av_c) + av_c + av * av_c)
    @test all(@. exp(av) + av + av * av ≈ exp(av_c) + av_c + av * av_c)
    av2 = PatternVector(N, FillPattern(val))
    @test all(@. exp(av) + av + av2 ≈ exp(av_c) + av_c + av2)
    @test typeof((av2 .+ av).pattern) <: InitialValuePattern
    @test typeof((av33 .+ av).pattern) <: InitialValuePattern
    @test typeof((PatternVector(N, ZeroPattern(val)) .+ av).pattern) <: InitialValuePattern

    pattern_eo = PatternVectors.EvenOddPattern(0, 0)
    vec2 = PatternVector(N, pattern_eo)
    res = @. vec2 + av
    @test typeof(res.pattern) <: PaddedEvenOddPattern
    @test typeof((res .+ av).pattern) <: PaddedEvenOddPattern
    @test all(@. exp(av) + av + av2 ≈ exp(av_c) + av_c + av2)
end

@testset "FinalValuePattern" begin
    @test_throws "length of PatternVector for pattern" PatternVector(0, FinalValuePattern(0, 0))
    N = 11
    val = 0.0
    av = PatternVector(N, PatternVectors.FinalValuePattern(val, val + 1))
    av33 = PatternVector(N, PatternVectors.FinalValuePattern(0, 1))
    @test PatternVectors.pattern_minimum_size(ZeroPattern(0)) == 1
    @test av[1] == val
    @test av[2] == val
    @test av[end] == val + 1
    Base.showarg(Core.CoreSTDOUT(), av, nothing)
    println(av)
    @show av
    av_c = collect(av)
    @test sum(av) ≈ sum(av_c)
    @test typeof(av[1:2]) <: PatternVector
    @test typeof(av[:]) <: PatternVector
    @test typeof(1 .+ av) <: PatternVector
    @test typeof(av .+ 1) <: PatternVector
    @test typeof(sin.(av)) <: PatternVector
    @test !(typeof(av .+ tuple(ones(N)...)) <: PatternVector)
    @test !(typeof(tuple(ones(N)...) .+ av) <: PatternVector)
    @test typeof(@. sin(av) * av + 1 + av) <: PatternVector
    @test typeof(@. sin(av) * av + 1 + exp(av)) <: PatternVector
    @test typeof(@. sin(av) * av * av + 1 + exp(av)) <: PatternVector
    @test typeof(@. sin(cos(av)) * av * av + exp(1) + exp(av)) <: PatternVector
    @test typeof(@. 2 + sin(av) * av + 1 + exp(av)) <: PatternVector
    @test all(@. av ≈ av_c)
    @test all(@. sin(av) ≈ sin(av_c))
    @test all(@. exp(av) + av ≈ exp(av_c) + av_c)
    @test all(@. exp(av) + av_c ≈ exp(av_c) + av)
    @test all(@. exp(av) + av_c ≈ exp(av_c) + av_c)
    @test all(@. exp(av) + av_c + av * av_c ≈ exp(av_c) + av_c + av * av_c)
    @test all(@. exp(av) + av + av * av ≈ exp(av_c) + av_c + av * av_c)
    av2 = PatternVector(N, FillPattern(val))
    @test all(@. exp(av) + av + av2 ≈ exp(av_c) + av_c + av2)
    @test typeof((av2 .+ av).pattern) <: FinalValuePattern
    @test typeof((av33 .+ av).pattern) <: FinalValuePattern
    @test typeof((PatternVector(N, ZeroPattern(val)) .+ av).pattern) <: FinalValuePattern
    @test typeof((PatternVector(N, InitialValuePattern(val, val + 1)) .+ av).pattern) <: PaddedFillPattern
    @test typeof((PatternVector(N, EvenOddPattern(val, val + 1)) .+ av).pattern) <: PaddedEvenOddPattern

    pattern_eo = PatternVectors.EvenOddPattern(0, 0)
    vec2 = PatternVector(N, pattern_eo)
    res = @. vec2 + av
    @test typeof(res.pattern) <: PaddedEvenOddPattern
    @test typeof((res .+ av).pattern) <: PaddedEvenOddPattern
    @test all(@. exp(av) + av + av2 ≈ exp(av_c) + av_c + av2)
end

@testset "PaddedFillPattern" begin
    @test_throws "length of PatternVector for pattern" PatternVector(0, PaddedFillPattern(0, 0, 0))
    N = 11
    val = 0.0
    av = PatternVector(N, PatternVectors.PaddedFillPattern(val, val + 1, val + 2))
    av33 = PatternVector(N, PatternVectors.PaddedFillPattern(0, 1, 2))
    @test PatternVectors.pattern_minimum_size(ZeroPattern(0)) == 1
    @test av[1] == val
    @test av[2] == val + 1
    @test av[end] == val + 2
    Base.showarg(Core.CoreSTDOUT(), av, nothing)
    println(av)
    @show av
    av_c = collect(av)
    @test sum(av) ≈ sum(av_c)
    @test typeof(av[1:3]) <: PatternVector
    @test typeof(av[:]) <: PatternVector
    @test typeof(1 .+ av) <: PatternVector
    @test typeof(av .+ 1) <: PatternVector
    @test typeof(sin.(av)) <: PatternVector
    @test !(typeof(av .+ tuple(ones(N)...)) <: PatternVector)
    @test !(typeof(tuple(ones(N)...) .+ av) <: PatternVector)
    @test typeof(@. sin(av) * av + 1 + av) <: PatternVector
    @test typeof(@. sin(av) * av + 1 + exp(av)) <: PatternVector
    @test typeof(@. sin(av) * av * av + 1 + exp(av)) <: PatternVector
    @test typeof(@. sin(cos(av)) * av * av + exp(1) + exp(av)) <: PatternVector
    @test typeof(@. 2 + sin(av) * av + 1 + exp(av)) <: PatternVector
    @test all(@. av ≈ av_c)
    @test all(@. sin(av) ≈ sin(av_c))
    @test all(@. exp(av) + av ≈ exp(av_c) + av_c)
    @test all(@. exp(av) + av_c ≈ exp(av_c) + av)
    @test all(@. exp(av) + av_c ≈ exp(av_c) + av_c)
    @test all(@. exp(av) + av_c + av * av_c ≈ exp(av_c) + av_c + av * av_c)
    @test all(@. exp(av) + av + av * av ≈ exp(av_c) + av_c + av * av_c)
    av2 = PatternVector(N, FillPattern(val))
    @test all(@. exp(av) + av + av2 ≈ exp(av_c) + av_c + av2)
    @test typeof((av2 .+ av).pattern) <: PaddedFillPattern
    @test typeof((av33 .+ av).pattern) <: PaddedFillPattern
    @test typeof((PatternVector(N, ZeroPattern(val)) .+ av).pattern) <: PaddedFillPattern
    @test typeof((PatternVector(N, InitialValuePattern(val, val + 1)) .+ av).pattern) <: PaddedFillPattern
    @test typeof((PatternVector(N, FinalValuePattern(val, val + 1)) .+ av).pattern) <: PaddedFillPattern
    @test typeof((PatternVector(N, EvenOddPattern(val, val + 1)) .+ av).pattern) <: PaddedEvenOddPattern

    pattern_eo = PatternVectors.EvenOddPattern(0, 0)
    vec2 = PatternVector(N, pattern_eo)
    res = @. vec2 + av
    @test typeof(res.pattern) <: PaddedEvenOddPattern
    @test typeof((res .+ av).pattern) <: PaddedEvenOddPattern
    @test all(@. exp(av) + av + av2 ≈ exp(av_c) + av_c + av2)
end

@testset "PaddedEvenOddPattern" begin
    N = 11
    av = PatternVector(N, PaddedEvenOddPattern(-2.0, 3.0, 2.0, 4.0))
    @test PatternVectors.pattern_minimum_size(PaddedEvenOddPattern(0.0, 0.0, 0.0, 0.0)) == 4
    @test_throws DomainError PatternVector(3, PaddedEvenOddPattern(0.0, 0.0, 0.0, 0.0))
    @test_throws DomainError av[1:3]
    @test av[1] == -2.0
    @test av[2] == 3.0
    @test av[3] == 2.0
    @test av[4] == 3.0
    @test av[end] == 4.0
    @show av
    Base.showarg(Core.CoreSTDOUT(), av, nothing)
    println(av)
    av_c = collect(av)
    @test typeof(av[1:1:5]) <: AlternatePaddedVector_test
    @test typeof(av[:]) <: AlternatePaddedVector_test
    @test typeof(1 .+ av) <: AlternatePaddedVector_test
    @test typeof(av .+ 1) <: AlternatePaddedVector_test
    @test typeof(sin.(av)) <: AlternatePaddedVector_test
    @test typeof(@. sin(av) * av + 1 + av) <: AlternatePaddedVector_test
    @test typeof(@. sin(av) * av + 1 + exp(av)) <: AlternatePaddedVector_test
    @test typeof(@. sin(av) * av * av + 1 + exp(av)) <: AlternatePaddedVector_test
    @test typeof(@. 2 + sin(av) * av + 1 + exp(av)) <: AlternatePaddedVector_test
    @test typeof(@. 2 + sin(av) * av + 1 + log(abs(av))) <: AlternatePaddedVector_test
    @test typeof(@. 2 + sin(cos(av)) * av + 1 + exp(av) + exp(1)) <: AlternatePaddedVector_test
    @test !(typeof(av .+ tuple(ones(N)...)) <: AlternatePaddedVector_test)
    @test !(typeof(tuple(ones(N)...) .+ av) <: AlternateVector_test)
    @test all(@. av ≈ av_c)
    @test all(@. av[1:5] ≈ av_c[1:5])
    @test all(@. av[3:2:9] ≈ av_c[3:2:9])
    @test all(@. av[1:1:9] ≈ av_c[1:1:9])
    @test all(@. av[4:9] ≈ av_c[4:9])
    @test all(@. sin(av) ≈ sin(av_c))
    @test all(@. exp(av) + av ≈ exp(av_c) + av_c)
    @test all(@. exp(av) + av + 2 ≈ exp(av_c) + av_c + 2)
    @test all(@. exp(av) + av_c ≈ exp(av_c) + av)
    @test all(@. exp(av) + av_c ≈ exp(av_c) + av_c)
    @test all(@. exp(av) + av_c + av * av_c ≈ exp(av_c) + av_c + av * av_c)
    @test all(@. (exp(av) * av_c) + (av * av_c) ≈ exp(av_c) * av_c + av * av_c)
    @test all(@. exp(av) + av + av * av ≈ exp(av_c) + av_c + av * av_c)
    @test all(@. exp(av + sin(1 + av)) + av + av * av ≈ exp(av_c + sin(1 + av_c)) + av_c + av * av_c)
    @test sum(av) ≈ sum(av_c)
    av_d = PatternVector(N, PaddedEvenOddPattern(Dates.Date(1992, 1, 1), Dates.Date(1992, 9, 1), Dates.Date(1922, 1, 1), Dates.Date(1392, 10, 1)))
    av_d_1 = collect(av_d)
    one_day = Dates.Day(1)
    res_av_with_ref = @. av_d + one_day
    @test all(@. res_av_with_ref == av_d_1 + one_day)
    #mixture
    av_1 = PatternVector(N, EvenOddPattern(-2.0, 3.0))
    @test typeof(av_1 .+ av) <: AlternatePaddedVector_test
    #test sparse
    sparse_p = spzeros(Float64, N)
    sparse_p[1] = 4.0
    @test typeof(sin.(av .* sparse_p)) <: typeof(sparse_p)

    #test lazy arrays
    D = LazyArray(@~ exp.(1:N))
    D_c = collect(D)
    res_mul_lazy_1 = D .* av
    res_mul_lazy_2 = av .* D
    res_mul = av_c .* D_c
    @test all(@. res_mul_lazy_1 ≈ res_mul)
    @test all(@. res_mul_lazy_2 ≈ res_mul)
    res_mul_lazy_1 = @. exp(av * av_c) + av * D * av * av_c + D * av
    res_mul_lazy_2 = @. exp(av_c * av_c) + av_c * D * av_c * av_c + D * av_c
    @test all(@. res_mul_lazy_1 ≈ res_mul_lazy_2)

    x_zero_dim = ZeroDimensionalVector(1)
    res_lazy_1_zero_dim = x_zero_dim .* av
    res_lazy_2_zero_dim = av .* x_zero_dim
    @test all(@. res_lazy_1_zero_dim ≈ res_lazy_2_zero_dim)
    res_lazy_1_zero_dim = @. (av_c * av) * x_zero_dim
    res_lazy_2_zero_dim = @. av_c * av * x_zero_dim
    @test all(@. res_lazy_1_zero_dim ≈ res_lazy_2_zero_dim)

    x_zero_dim_r = Array{Float64, 0}(undef)
    x_zero_dim_r .= 0
    res_zero_d_r = @. x_zero_dim_r * av
    res_zero_d = @. x_zero_dim_r * av_c
    @test all(@. res_zero_d ≈ res_zero_d_r)

    av_int = PatternVector(N, PaddedEvenOddPattern(-2, 3, 2, 4))
    @test all(@. exp(av) + av_int ≈ exp(av_c) + av_int)
    @test all(@. exp(av) + av_int ≈ av_int + exp(av))
end

@testset "Mixtures" begin
    N = 11
    av = PatternVector(N, PaddedEvenOddPattern(-2.0, 3.0, 2.0, 4.0))
    @test_throws DomainError av[1:3]
    @test av[1] == -2.0
    @test av[2] == 3.0
    @test av[3] == 2.0
    @test av[4] == 3.0
    @test av[end] == 4.0
    @show av
    Base.showarg(Core.CoreSTDOUT(), av, nothing)
    println(av)
    av_c = collect(av)
    #mixture
    av_1 = PatternVector(N, EvenOddPattern(-2.0, 3.0))
    pattern_zero = PatternVectors.ZeroPattern(0.0)
    vec_zero = PatternVector(N, pattern_zero)
    @test typeof(av_1 .+ vec_zero) <: AlternateVector_test
    pattern_ones = PatternVectors.FillPattern(1.0)
    vec_one = PatternVector(N, pattern_ones)
    @test typeof(vec_zero .+ vec_one) <: typeof(vec_one)
    @test typeof(vec_one .+ vec_zero) <: typeof(vec_one)
    @test typeof(av_1 .+ vec_one) <: AlternateVector_test
    @test typeof(av_1 .+ vec_zero) <: AlternateVector_test
    @test typeof(vec_zero .+ av_1) <: AlternateVector_test
    @test typeof(vec_one .+ av) <: AlternatePaddedVector_test
    @test typeof(vec_zero .+ av) <: AlternatePaddedVector_test
    @test typeof(vec_zero .+ av .+ vec_one .+ vec_zero .+ av_1) <: AlternatePaddedVector_test
    @test !(typeof(vec_zero .+ vec_one .+ vec_zero .+ av_1) <: AlternatePaddedVector_test)
    @test !(typeof(vec_zero .+ vec_one .+ vec_zero) <: AlternatePaddedVector_test)
    @test !(typeof(av_1 .+ vec_one) <: AlternatePaddedVector_test)
    @test !(typeof(av_1 .+ vec_zero) <: AlternatePaddedVector_test)

    @test typeof(av_1 .+ av) <: AlternatePaddedVector_test
    #test sparse
    sparse_p = spzeros(Float64, N)
    sparse_p[1] = 4.0
    @test typeof(sin.(av .* sparse_p)) <: typeof(sparse_p)

    #test lazy arrays
    D = LazyArray(@~ exp.(1:N))
    D_c = collect(D)
    res_mul_lazy_1 = D .* av
    res_mul_lazy_2 = av .* D
    res_mul = av_c .* D_c
    @test all(@. res_mul_lazy_1 ≈ res_mul)
    @test all(@. res_mul_lazy_2 ≈ res_mul)
    res_mul_lazy_1 = @. exp(av * av_c) + av * D * av * av_c + D * av
    res_mul_lazy_2 = @. exp(av_c * av_c) + av_c * D * av_c * av_c + D * av_c
    @test all(@. res_mul_lazy_1 ≈ res_mul_lazy_2)

    x_zero_dim = ZeroDimensionalVector(1)
    res_lazy_1_zero_dim = x_zero_dim .* av
    res_lazy_2_zero_dim = av .* x_zero_dim
    @test all(@. res_lazy_1_zero_dim ≈ res_lazy_2_zero_dim)
    res_lazy_1_zero_dim = @. (av_c * av) * x_zero_dim
    res_lazy_2_zero_dim = @. av_c * av * x_zero_dim
    @test all(@. res_lazy_1_zero_dim ≈ res_lazy_2_zero_dim)

    x_zero_dim_r = Array{Float64, 0}(undef)
    x_zero_dim_r .= 0
    res_zero_d_r = @. x_zero_dim_r * av
    res_zero_d = @. x_zero_dim_r * av_c
    @test all(@. res_zero_d ≈ res_zero_d_r)
end

@testset "SparseArraysExt for EvenOddPattern" begin
    N = 11
    pattern_iv = EvenOddPattern(-2.0, 3.0)
    av = PatternVector(N, pattern_iv)
    av_c = collect(av)
    #test sparse
    sparse_p = spzeros(Float64, N)
    sparse_p[1] = 4.0
    @test typeof(sin.(av .* sparse_p)) <: typeof(sparse_p)
    @test all(@. av * sparse_p ≈ av_c * sparse_p)
end

@testset "SparseArraysExt for PaddedEvenOddPattern" begin
    N = 11
    av = PatternVector(N, PaddedEvenOddPattern(-2.0, 3.0, 2.0, 4.0))
    av_c = collect(av)
    #test sparse
    sparse_p = spzeros(Float64, N)
    sparse_p[1] = 4.0
    @test typeof(sin.(av .* sparse_p)) <: typeof(sparse_p)
    @test all(@. av * sparse_p ≈ av_c * sparse_p)
end

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

@testset "Composite Broadcasting for EvenOddPattern" begin
    N = 11
    pattern_iv = EvenOddPattern(-2.0, 4.0)
    av = PatternVector(N, pattern_iv)
    global incr = 0
    function scalar_f_av(x)
        global incr += 1
        return sin(x)
    end
    res = @. scalar_f_av(av)
    @test incr == 2
    incr = 0
    av_c = collect(av)
    res = @. scalar_f_av(av) * av_c
    @test incr == 2
end

@testset "Composite Broadcasting for PaddedEvenOddPattern" begin
    N = 11
    pattern_iv = PaddedEvenOddPattern(-2.0, 3.0, 2.0, 4.0)
    av = PatternVector(N, pattern_iv)
    global incr = 0
    function scalar_f(x)
        global incr += 1
        return sin(x)
    end
    res = @. scalar_f(av)
    @test incr == 4
    incr = 0
    av_c = collect(av)
    res = @. scalar_f(av) * av_c
    @test incr == 4
end
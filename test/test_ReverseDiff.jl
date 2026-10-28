using PatternVectors, Test, ReverseDiff

@testset "copy of a mixed Broadcasted" begin
    N = 11
    av = PatternVector(N, EvenOddPattern(1.0, -2.0))
    x = collect(1.0:N)
    bc = Broadcast.instantiate(Broadcast.broadcasted(*, av, x))
    @test bc isa Broadcast.Broadcasted{<:PatternVectors.PatternMixtureArrayStyle}
    res = copy(bc)
    @test res isa Vector{Float64}
    @test res == collect(av) .* x
end

@testset "ReverseDiff broadcast style" begin
    av = PatternVector(11, EvenOddPattern(1.0, -2.0))
    tracked_vector = ReverseDiff.track(ones(11))
    tracked_scalar = ReverseDiff.track(1.0)
    @test Broadcast.combine_styles(av, tracked_vector) isa ReverseDiff.TrackedStyle{1}
    @test Broadcast.combine_styles(tracked_scalar, av) isa ReverseDiff.TrackedStyle{1}
    @test Broadcast.combine_styles(av, ones(11, 2), tracked_scalar) isa ReverseDiff.TrackedStyle{2}
end

@testset "ReverseDiff gradients" begin
    N = 11
    av = PatternVector(N, PaddedEvenOddPattern(1.0, 2.0, -3.0, 4.0))
    av_c = collect(av)
    x = collect(range(0.1, 1.3, length = N))
    functions = (
        # tracked array, real result: ReverseDiff's own broadcast rule
        (pv, y) -> sum(pv .* sin.(y)),
        # tracked scalar
        (pv, y) -> sum(@. exp(pv * y[1]) + pv),
        # PatternVector, dense array and tracked array in the same broadcast
        (pv, y) -> sum(pv .* ones(N) .* y .^ 2),
        # complex result: ReverseDiff falls back to copying the untracked Broadcasted
        (pv, y) -> sum(real.(pv .* cis.(y))),
        (pv, y) -> sum(abs2.(@. pv * exp(y * im) + 1)),
    )
    for f in functions
        @test ReverseDiff.gradient(y -> f(av, y), x) ≈ ReverseDiff.gradient(y -> f(av_c, y), x)
    end
end

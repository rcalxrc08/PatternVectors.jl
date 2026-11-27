# PatternVectors compiled by Reactant (XLA): values against plain Julia, and the structure of the XLA program, which
# must hold the pattern values only, never a dense array of n values. Not run on CI (Reactant takes minutes to
# precompile); run it locally after a change to ext/ReactantExt.jl, src/02_broadcast or src/04_patterns:
#     julia --project=test/reactant -e 'import Pkg; Pkg.instantiate(); include("test/reactant/runtests.jl")'
# The XLA backend comes from the environment variable REACTANT_BACKEND ("cpu" by default, "cuda" on Linux with a GPU).
using Test, Reactant, Enzyme, PatternVectors, LinearAlgebra

const BACKEND = get(ENV, "REACTANT_BACKEND", "cpu")
Reactant.set_default_backend(BACKEND)

# Above 1024 elements Reactant keeps a constant as it is, so a dense array of the n values would show up.
const n = 2001
const PATTERNS = (FillPattern(0.5), ZeroPattern(0.0), EvenOddPattern(0.5, 2.0), InitialValuePattern(0.25, 1.0), FinalValuePattern(1.0, 0.25), PaddedFillPattern(0.5, 1.0, 0.5), PaddedEvenOddPattern(1 / 3, 4 / 3, 2 / 3, 1 / 3))

# The largest constant of the program after Reactant's optimizations: MLIR prints dense constants as a list or in hex.
function largest_constant(f, args...)
    hlo = string(@code_hlo optimize = true f(args...))
    sizes = [prod(parse.(Int, split(m.captures[1], 'x'))) for m in eachmatch(r"stablehlo\.constant dense<(?:\"0x|\[)[^>]*> : tensor<([\dx]+)x", hlo)]
    return isempty(sizes) ? 0 : maximum(sizes)
end
# The same for the program optimized by XLA (it folds some operations on constants)
function largest_xla_constant(f, args...)
    xla = string(@code_xla f(args...))
    sizes = [prod(parse.(Int, split(m.captures[1], ','))) for m in eachmatch(r"\[([\d,]+)\]\{[\d,]*\} constant\(", xla)]
    return isempty(sizes) ? 0 : maximum(sizes)
end
length_n_tensors(f, args...) = count("tensor<$(n)x", string(@code_hlo optimize = true f(args...)))

tonumber(x) = x isa Number ? Float64(x) : only(Array(x))

# Pattern values from a traced array, to build a PatternVector inside the compiled function
traced_pattern(a) = PatternVector(n, PaddedEvenOddPattern(Reactant.@allowscalar(a[1]), Reactant.@allowscalar(a[2]), Reactant.@allowscalar(a[3]), Reactant.@allowscalar(a[4])))
mixed(a, x) = traced_pattern(a) .* x
weighted(a, x) = sum(traced_pattern(a) .* exp.(x))
pattern_only(a) = sum(traced_pattern(a) .* traced_pattern(a) .+ 1)
dotted(a, x) = dot(traced_pattern(a), x)
int_index(a) = traced_pattern(a)[3]
traced_index(a, i) = traced_pattern(a)[i]
ranged(a) = sum(traced_pattern(a)[2:7])
cumulative(a) = cumsum(traced_pattern(a))
collected(a) = collect(traced_pattern(a))

@testset "PatternVectors on Reactant ($BACKEND)" begin
    x = collect(range(-1.0, 1.0; length = n))
    xr = Reactant.to_rarray(x)
    @testset "constant $(nameof(typeof(p)))" for p in PATTERNS
        w = PatternVector(n, p)
        product(x) = w .* exp.(x)
        quadrature(x) = sum(w .* exp.(x))
        @test Array(@jit(product(xr))) ≈ w .* exp.(x) rtol = 1e-14
        @test tonumber(@jit(quadrature(xr))) ≈ sum(w .* exp.(x)) rtol = 1e-13
        @test largest_constant(quadrature, xr) < n
        @test largest_xla_constant(quadrature, xr) < n
    end
    @testset "Float32" begin
        w = PatternVector(n, PaddedEvenOddPattern(1.0f0 / 3, 4.0f0 / 3, 2.0f0 / 3, 1.0f0 / 3))
        quadrature32(x) = sum(w .* exp.(x))
        x32 = Float32.(x)
        out = @jit(quadrature32(Reactant.to_rarray(x32)))
        @test Reactant.unwrapped_eltype(typeof(out)) == Float32
        @test tonumber(out) ≈ sum(w .* exp.(x32)) rtol = 1e-5
    end
    @testset "traced pattern values" begin
        ab = [1 / 3, 4 / 3, 2 / 3, 0.25]
        abr = Reactant.to_rarray(ab)
        w = PatternVector(n, PaddedEvenOddPattern(ab...))
        @test Array(@jit(mixed(abr, xr))) ≈ w .* x
        @test tonumber(@jit(weighted(abr, xr))) ≈ sum(w .* exp.(x)) rtol = 1e-13
        @test tonumber(@jit(dotted(abr, xr))) ≈ dot(w, x) rtol = 1e-12 atol = 1e-12
        # only the pattern values are computed: no tensor of length n in the program
        @test tonumber(@jit(pattern_only(abr))) ≈ sum(w .* w .+ 1)
        @test length_n_tensors(pattern_only, abr) == 0
        @test tonumber(@jit(int_index(abr))) == w[3]
        @test tonumber(@jit(traced_index(abr, Reactant.to_rarray(n; track_numbers = Number)))) == w[n]
        @test tonumber(@jit(ranged(abr))) ≈ sum(w[2:7])
        @test Array(@jit(cumulative(abr))) ≈ cumsum(w)
        @test Array(@jit(collected(abr))) == collect(w)
    end
    @testset "Enzyme" begin
        ab = [1 / 3, 4 / 3, 2 / 3, 0.25]
        abr = Reactant.to_rarray(ab)
        w = PatternVector(n, PaddedEvenOddPattern(ab...))
        e = exp.(x)
        expected = [e[1], sum(e[2:2:(n-1)]), sum(e[3:2:(n-1)]), e[n]]
        @test Array(@jit(Enzyme.gradient(Reverse, weighted, abr, Const(xr)))[1]) ≈ expected rtol = 1e-13
        @test Array(@jit(Enzyme.gradient(Reverse, weighted, Const(abr), xr))[2]) ≈ w .* e rtol = 1e-14
        tangent = Reactant.to_rarray([1.0, 0.0, 0.0, 0.0])
        @test tonumber(only(@jit(Enzyme.autodiff(Forward, weighted, Duplicated(abr, tangent), Const(xr))))) ≈ e[1] rtol = 1e-14
    end
end

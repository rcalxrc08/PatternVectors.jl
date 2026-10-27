# This file was generated, do not modify it. # hide
using PatternVectors, BenchmarkTools, LinearAlgebra
n = 10_000
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
# This file was generated, do not modify it. # hide
using PatternVectors, BenchmarkTools
n=10_000
x=PatternVector(n,EvenOddPattern(0.2,2.3))
y=randn(n)
x_c=collect(x)
@btime @. $x*$y;
@btime @. $x_c*$y;
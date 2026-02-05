# This file was generated, do not modify it. # hide
using PatternVectors
x=PatternVector(10,EvenOddPattern(0.2,2.3))
y=randn(10)
z=PatternVector(10,PaddedEvenOddPattern(0.2,-2.0,4.0,2.3))
@. sin(x)*y+z
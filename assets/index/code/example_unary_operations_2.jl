# This file was generated, do not modify it. # hide
using PatternVectors
pattern=EvenOddPattern(0.2,-2.0)
av=PatternVector(10,pattern)
z_sin=@. sin(av)
z_sin
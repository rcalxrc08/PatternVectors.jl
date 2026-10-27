# This file was generated, do not modify it. # hide
using PatternVectors
simpson(f_x, h) = h * sum(PatternVector(length(f_x), PaddedEvenOddPattern(1/3, 4/3, 2/3, 1/3)) .* f_x)
h = 0.01
simpson(exp.(0:h:1), h) - (exp(1) - 1)
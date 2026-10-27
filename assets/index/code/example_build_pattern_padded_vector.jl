# This file was generated, do not modify it. # hide
using PatternVectors
initial_value, value_even, value_odd, final_value = 0.2, 2.3, -0.2, 1.3
x_av = PatternVector(7, PaddedEvenOddPattern(initial_value, value_even, value_odd, final_value))
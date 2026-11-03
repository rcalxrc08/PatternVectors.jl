using LinearAlgebra

# dot with broadcasts and a reduction only, so that it runs on the device of a dense argument: the generic dot of
# LinearAlgebra iterates both vectors element by element, which is scalar indexing on a GPU array. It is reached in
# reverse mode, where the rrule of sum returns a PatternVector cotangent and ChainRules computes the cotangent of a
# scalar times an array as dot(Ȳ, B).
function check_dot_lengths(x, y)
    length(x) == length(y) || throw(DimensionMismatch("first array has length $(length(x)) which does not match the length of the second, $(length(y))."))
    return nothing
end
function pattern_dot(x, y)
    check_dot_lengths(x, y)
    return sum(dot.(x, y))
end
LinearAlgebra.dot(x::PatternVector, y::AbstractVector) = pattern_dot(x, y)
LinearAlgebra.dot(x::AbstractVector, y::PatternVector) = pattern_dot(x, y)
LinearAlgebra.dot(x::PatternVector, y::PatternVector) = pattern_dot(x, y)

# A fill needs a single reduction of the other vector.
const FillPatternVector{T} = PatternVector{T, FillPattern{T}}
function LinearAlgebra.dot(x::FillPatternVector, y::AbstractVector)
    check_dot_lengths(x, y)
    return dot(x.pattern.value, sum(y))
end
function LinearAlgebra.dot(x::AbstractVector, y::FillPatternVector)
    check_dot_lengths(x, y)
    return dot(sum(x), y.pattern.value)
end
LinearAlgebra.dot(x::FillPatternVector, y::PatternVector) = invoke(dot, Tuple{FillPatternVector, AbstractVector}, x, y)
LinearAlgebra.dot(x::PatternVector, y::FillPatternVector) = invoke(dot, Tuple{AbstractVector, FillPatternVector}, x, y)
LinearAlgebra.dot(x::FillPatternVector, y::FillPatternVector) = invoke(dot, Tuple{FillPatternVector, AbstractVector}, x, y)

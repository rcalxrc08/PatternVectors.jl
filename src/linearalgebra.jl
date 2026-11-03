using LinearAlgebra

# dot with broadcasts and a reduction only, so that it runs on the device of a dense argument: the generic dot of
# LinearAlgebra iterates both vectors element by element, which is scalar indexing on a GPU array. It is reached in
# reverse mode, where the rrule of sum returns a PatternVector cotangent and ChainRules computes the cotangent of a
# scalar times an array as dot(Ȳ, B).
# Three methods only, with the patterns dispatched inside: every method of dot on a pair of types can be ambiguous
# with those of other packages (ReverseDiff defines dot(::AbstractVector, ::TrackedArray), see ReverseDiffExt).
function check_dot_lengths(x, y)
    length(x) == length(y) || throw(DimensionMismatch("first array has length $(length(x)) which does not match the length of the second, $(length(y))."))
    return nothing
end
const FillPatternVector{T} = PatternVector{T, FillPattern{T}}
pattern_dot(x::PatternVector, y) = sum(dot.(x, y))
# a fill needs a single reduction of the other vector
pattern_dot(x::FillPatternVector, y) = dot(x.pattern.value, sum(y))

function LinearAlgebra.dot(x::PatternVector, y::AbstractVector)
    check_dot_lengths(x, y)
    return pattern_dot(x, y)
end
function LinearAlgebra.dot(x::AbstractVector, y::PatternVector)
    check_dot_lengths(x, y)
    return conj(pattern_dot(y, x))
end
function LinearAlgebra.dot(x::PatternVector, y::PatternVector)
    check_dot_lengths(x, y)
    return pattern_dot(x, y)
end

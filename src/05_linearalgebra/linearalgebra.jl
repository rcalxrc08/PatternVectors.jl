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

# Reverse rules of dot (Zygote): the weighted sums of quadratures, dot(w, y) with constant weights w, avoid
# materializing w .* y, and the cotangent of the dense argument is again a PatternVector, O(1) and isbits (#21).
# Convention of ChainRules' rule for dot: x̄ = y .* conj(Δ), ȳ = x .* Δ. The cotangents are thunks, computed only if
# used; integer and Bool vectors (constant weights) get none. Δ is unthunked: Zygote ≥ 0.7.17 defers it (#19).
# The ChainRules rule for any two arrays would give the same values: measured at n = 10^6, it costs as much as
# sum(w .* y) or more, while this one allocates 1.3-1.75 times less on the GPU and runs 1.2-1.5 times faster there.
has_no_tangent(x) = eltype(x) <: Union{Integer, Bool}
function dot_rrule(x, y)
    function dot_pullback(Δ)
        Δu = unthunk(Δ)
        x̄ = has_no_tangent(x) ? NoTangent() : @thunk(y .* conj(Δu))
        ȳ = has_no_tangent(y) ? NoTangent() : @thunk(x .* Δu)
        return NoTangent(), x̄, ȳ
    end
    return dot(x, y), dot_pullback
end
ChainRulesCore.rrule(::typeof(dot), x::PatternVector, y::AbstractVector) = dot_rrule(x, y)
ChainRulesCore.rrule(::typeof(dot), x::AbstractVector, y::PatternVector) = dot_rrule(x, y)
ChainRulesCore.rrule(::typeof(dot), x::PatternVector, y::PatternVector) = dot_rrule(x, y)

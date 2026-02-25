function ChainRulesCore.rrule(::Type{PatternVector}, n::Int64, pattern::P) where {P}
    function PatternVector_pb(Δapv)
        NoTangent(), NoTangent(), pattern_to_vector_pullback(P, Δapv, n)
    end
    return PatternVector(n, pattern), PatternVector_pb
end

function ChainRulesCore.rrule(::typeof(Base.sum), x::AbstractArray{T, 1}) where {T}
    function sum_pb(Δapv)
        pattern_fill = FillPattern(Δapv)
        NoTangent(), PatternVector(length(x), pattern_fill)
    end
    return sum(x), sum_pb
end
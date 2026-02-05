using ChainRulesCore

function ChainRulesCore.rrule(::Type{PatternVector}, n::Int64, pattern::P) where {P}
    function PatternVector_pb(Δapv)
        NoTangent(), NoTangent(), pattern_to_vector_pullback(P, Δapv, n)
    end
    return PatternVector(n, pattern), PatternVector_pb
end

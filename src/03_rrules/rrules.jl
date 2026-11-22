# The pullbacks unthunk their cotangent: Zygote ≥ 0.7.17 passes the cotangent of a scalar as a deferred Thunk, which a
# pattern (or a PatternVector of thunks, unthunked by Zygote with map into a host Vector) must not hold (#19).
function ChainRulesCore.rrule(::Type{PatternVector}, n::Int64, pattern::P) where {P}
    function PatternVector_pb(Δapv)
        NoTangent(), NoTangent(), pattern_to_vector_pullback(P, unthunk(Δapv), n)
    end
    return PatternVector(n, pattern), PatternVector_pb
end

function ChainRulesCore.rrule(::typeof(Base.sum), x::AbstractArray{T, 1}) where {T}
    function sum_pb(Δapv)
        pattern_fill = FillPattern(unthunk(Δapv))
        NoTangent(), PatternVector(length(x), pattern_fill)
    end
    return sum(x), sum_pb
end

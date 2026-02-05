
## EvenOddPattern
# EvenOddPattern represents a pattern that alternates between two values for odd and even indices.
struct EvenOddPattern{T} <: AbstractPattern{T}
    value_odd::T
    value_even::T
    function EvenOddPattern(value_odd::T, value_even::T) where {T}
        return new{T}(value_odd, value_even)
    end
end

pattern_minimum_size(::Type{P}) where {P <: EvenOddPattern} = 2

function getindex_pattern(x::EvenOddPattern, ind::Int, ::Int)
    ifelse(isodd(ind), x.value_odd, x.value_even)
end

function getindex_pattern_range(x::P, el::AbstractRange{T}, n::Int) where {T <: Int, P <: EvenOddPattern}
    new_len = length(el)
    minimum_size = pattern_minimum_size(P)
    (minimum_size <= new_len) || throw("Trying to getindex with an AbstractRange of length $new_len. Provided length must be greater or equal to $minimum_size.")
    first_idx = el.start
    @views @inbounds odd_value = getindex_pattern(x, first_idx, n)
    @views @inbounds even_value = getindex_pattern(x, first_idx + step(el), n)
    return new_len, EvenOddPattern(odd_value, even_value)
end

function materialize_pattern(bc::Base.Broadcast.Broadcasted{ArrayStylePatternVector{P}, Nothing, <:F, <:R}) where {F, R, P <: EvenOddPattern}
    bc_f = Broadcast.flatten(bc)
    func = bc_f.f
    args = bc_f.args
    axes_result = Broadcast.combine_axes(args...)
    odd_part = func(flat_index.(args, 1)...)
    even_part = func(flat_index.(args, 2)...)
    return length(first(axes_result)), EvenOddPattern(odd_part, even_part)
end

# Function to determine the mixed pattern type when combining various patterns
determine_mixed_pattern(::Type{T}, ::Type{V}) where {T <: ZeroPattern{L}, V <: EvenOddPattern{N}} where {L, N} = EvenOddPattern{promote_type(L, N)}
determine_mixed_pattern(::Type{T}, ::Type{V}) where {T <: FillPattern{L}, V <: EvenOddPattern{N}} where {L, N} = EvenOddPattern{promote_type(L, N)}
determine_mixed_pattern(::Type{T}, ::Type{V}) where {T <: EvenOddPattern{L}, V <: EvenOddPattern{N}} where {L, N} = EvenOddPattern{promote_type(L, N)}

function ChainRulesCore.rrule(::Type{EvenOddPattern}, args...)
    function AbstractPattern_pb(Δapv)
        NoTangent(), (getfield(Δapv, arg) for arg in fieldnames(EvenOddPattern))...
    end
    return EvenOddPattern(args...), AbstractPattern_pb
end

function pattern_to_vector_pullback(::Type{P}, Δapv, n) where {P <: EvenOddPattern{T}} where {T}
    bd_val_v = PatternVector(n, EvenOddPattern(one(T), zero(T)))
    der_bound_initial_value = sum(Δapv .* bd_val_v)
    der_bound_final_value = sum(Δapv) - der_bound_initial_value
    return EvenOddPattern(der_bound_initial_value, der_bound_final_value)
end

#Since we use a sum function in the pullback, we implement it for EvenOddPattern
function Base.sum(x::PatternVector{T, P}) where {T, P <: EvenOddPattern{T}}
    isfinalodd = isodd(x.n)
    nhalf = div(x.n, 2)
    pattern = x.pattern
    return muladd(nhalf, pattern.value_even, (nhalf + isfinalodd) * pattern.value_odd)
end
## PaddedEvenOddPattern
# PaddedEvenOddPattern represents a pattern that alternates between two values for odd and even indices.
struct PaddedEvenOddPattern{T} <: AbstractPattern{T}
    bound_initial_value::T
    value_even::T
    value_odd::T
    bound_final_value::T
    function PaddedEvenOddPattern(bound_initial_value::T, value_even::T, value_odd::T, bound_final_value::T) where {T}
        return new{T}(bound_initial_value, value_even, value_odd, bound_final_value)
    end
end

pattern_minimum_size(::PaddedEvenOddPattern) = 4

function getindex_pattern(x::PaddedEvenOddPattern, ind::Int, n::Int)
    ifelse(ind == 1, x.bound_initial_value, ifelse(ind == n, x.bound_final_value, ifelse(isodd(ind), x.value_odd, x.value_even)))
end

function getindex_pattern_range(x::P, el::AbstractRange{T}, n::Int) where {T <: Int, P <: PaddedEvenOddPattern}
    new_len = length(el)
    minimum_size = pattern_minimum_size(x)
    (minimum_size <= new_len) || throw(DomainError(new_len, "Trying to getindex with an AbstractRange of length $new_len. Provided length must be greater or equal to $minimum_size."))
    first_idx = el.start
    new_len = length(el)
    @views @inbounds bound_initial_value = getindex_pattern(x, first_idx, n)
    step_el = step(el)
    next_idx = first_idx + step_el
    @views @inbounds even_value = getindex_pattern(x, next_idx, n) #This is a trick, in case n<4 it's breaking the bounds, but it's fine the function is well behaving anyway
    @views @inbounds odd_value = getindex_pattern(x, next_idx + step_el, n) #This is a trick, in case n<4 it's breaking the bounds, but it's fine the function is well behaving anyway
    @views @inbounds bound_final_value = getindex_pattern(x, last(el), n)

    return new_len, PaddedEvenOddPattern(bound_initial_value, even_value, odd_value, bound_final_value)
end

function materialize_pattern(bc::Base.Broadcast.Broadcasted{ArrayStylePatternVector{P}, Nothing, <:F, <:R}) where {F, R, P <: PaddedEvenOddPattern}
    bc_f = Broadcast.flatten(bc)
    func = bc_f.f
    args = bc_f.args
    axes_result = Broadcast.combine_axes(args...)
    n = length(first(axes_result))
    initial_part = func(flat_index.(args, 1)...)
    even_part = func(flat_index.(args, 2)...)
    odd_part = func(flat_index.(args, 3)...)
    end_part = func(flat_index.(args, n)...)
    return n, PaddedEvenOddPattern(initial_part, even_part, odd_part, end_part)
end

# Function to determine the mixed pattern type when combining various patterns
determine_mixed_pattern(::Type{T}, ::Type{V}) where {T <: ZeroPattern{M}, V <: PaddedEvenOddPattern{N}} where {M, N} = PaddedEvenOddPattern{promote_type(M, N)}
determine_mixed_pattern(::Type{T}, ::Type{V}) where {T <: EvenOddPattern{M}, V <: PaddedEvenOddPattern{N}} where {M, N} = PaddedEvenOddPattern{promote_type(M, N)}
determine_mixed_pattern(::Type{T}, ::Type{V}) where {T <: FillPattern{M}, V <: PaddedEvenOddPattern{N}} where {M, N} = PaddedEvenOddPattern{promote_type(M, N)}
determine_mixed_pattern(::Type{T}, ::Type{V}) where {T <: InitialValuePattern{M}, V <: PaddedEvenOddPattern{N}} where {M, N} = PaddedEvenOddPattern{promote_type(M, N)}
determine_mixed_pattern(::Type{T}, ::Type{V}) where {T <: PaddedEvenOddPattern{M}, V <: PaddedEvenOddPattern{N}} where {M, N} = PaddedEvenOddPattern{promote_type(M, N)}

function ChainRulesCore.rrule(::Type{PaddedEvenOddPattern}, args...)
    function AbstractPattern_pb(Δapv)
        NoTangent(), (getfield(Δapv, arg) for arg in fieldnames(PaddedEvenOddPattern))...
    end
    return PaddedEvenOddPattern(args...), AbstractPattern_pb
end

function pattern_to_vector_pullback(::Type{P}, Δapv, n) where {P <: PaddedEvenOddPattern{T}} where {T}
    bd_val_v = PatternVector(n, PaddedEvenOddPattern(one(T), zero(T), zero(T), zero(T)))
    der_bound_initial_value = sum(Δapv .* bd_val_v)
    odd_v = PatternVector(n, PaddedEvenOddPattern(zero(T), zero(T), one(T), zero(T)))
    odd_der = sum(odd_v .* Δapv)
    even_v = PatternVector(n, PaddedEvenOddPattern(zero(T), one(T), zero(T), zero(T)))
    even_der = sum(even_v .* Δapv)
    der_bound_final_value = sum(Δapv) - even_der - odd_der - der_bound_initial_value
    return PaddedEvenOddPattern(der_bound_initial_value, even_der, odd_der, der_bound_final_value)
end

#Since we use a sum function in the pullback, we implement it for PaddedEvenOddPattern
function Base.sum(x::PatternVector{T, P}) where {T, P <: PaddedEvenOddPattern{T}}
    pattern = x.pattern
    isfinalodd = isodd(x.n)
    nhalf = div(x.n, 2) - 1
    return muladd(nhalf, pattern.value_odd, muladd(pattern.value_even, nhalf + isfinalodd, pattern.bound_initial_value + pattern.bound_final_value))
end
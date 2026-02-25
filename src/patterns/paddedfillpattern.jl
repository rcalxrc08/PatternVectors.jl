## PaddedFillPattern
# PaddedFillPattern represents a pattern that alternates between two values for odd and even indices.
struct PaddedFillPattern{T} <: AbstractPattern{T}
    bound_initial_value::T
    value::T
    bound_final_value::T
    function PaddedFillPattern(bound_initial_value::T, value::T, bound_final_value::T) where {T}
        return new{T}(bound_initial_value, value, bound_final_value)
    end
end

pattern_minimum_size(::PaddedFillPattern) = 3 # COV_EXCL_LINE

function getindex_pattern(x::PaddedFillPattern, ind::Int, n::Int)
    ifelse(ind == 1, x.bound_initial_value, ifelse(ind == n, x.bound_final_value, x.value))
end

function getindex_pattern_range(x::P, el::AbstractRange{T}, n::Int) where {T <: Int, P <: PaddedFillPattern}
    new_len = length(el)
    minimum_size = pattern_minimum_size(x)
    (minimum_size <= new_len) || throw(DomainError(new_len, "Trying to getindex with an AbstractRange of length $new_len. Provided length must be greater or equal to $minimum_size."))
    first_idx = el.start
    bound_initial_value = getindex_pattern(x, first_idx, n)
    step_el = step(el)
    next_idx = first_idx + step_el
    value = getindex_pattern(x, next_idx, n)
    bound_final_value = getindex_pattern(x, last(el), n)

    return new_len, PaddedFillPattern(bound_initial_value, value, bound_final_value)
end

function materialize_pattern(bc::Base.Broadcast.Broadcasted{ArrayStylePatternVector{P}, Nothing, <:F, <:R}) where {F, R, P <: PaddedFillPattern}
    bc_f = Broadcast.flatten(bc)
    func = bc_f.f
    args = bc_f.args
    axes_result = Broadcast.combine_axes(args...)
    n = length(first(axes_result))
    initial_part = func(flat_index.(args, 1)...)
    value_part = func(flat_index.(args, 2)...)
    end_part = func(flat_index.(args, n)...)
    return n, PaddedFillPattern(initial_part, value_part, end_part)
end

determine_mixed_pattern(::Type{T}, ::Type{V}) where {T <: PaddedFillPattern{M}, V <: PaddedFillPattern{N}} where {M, N} = PaddedFillPattern{promote_type(M, N)}

# Function to determine the mixed pattern type when combining various patterns
determine_mixed_pattern(::Type{T}, ::Type{V}) where {T <: ZeroPattern{M}, V <: PaddedFillPattern{N}} where {M, N} = PaddedFillPattern{promote_type(M, N)}
determine_mixed_pattern(::Type{T}, ::Type{V}) where {T <: FillPattern{M}, V <: PaddedFillPattern{N}} where {M, N} = PaddedFillPattern{promote_type(M, N)}
determine_mixed_pattern(::Type{T}, ::Type{V}) where {T <: InitialValuePattern{M}, V <: PaddedFillPattern{N}} where {M, N} = PaddedFillPattern{promote_type(M, N)}
determine_mixed_pattern(::Type{T}, ::Type{V}) where {T <: FinalValuePattern{M}, V <: PaddedFillPattern{N}} where {M, N} = PaddedFillPattern{promote_type(M, N)}

#Mixture generation
determine_mixed_pattern(::Type{T}, ::Type{V}) where {T <: InitialValuePattern{L}, V <: FinalValuePattern{N}} where {L, N} = PaddedFillPattern{promote_type(L, N)}

function ChainRulesCore.rrule(::Type{PaddedFillPattern}, bound_initial_value, value, bound_final_value)
    function AbstractPattern_pb(Δapv)
        NoTangent(), Δapv.bound_initial_value, Δapv.value, Δapv.bound_final_value
    end
    return PaddedFillPattern(bound_initial_value, value, bound_final_value), AbstractPattern_pb
end

function pattern_to_vector_pullback(::Type{P}, Δapv, n) where {P <: PaddedFillPattern{T}} where {T}
    bd_iv_v = PatternVector(n, InitialValuePattern(one(T), zero(T)))
    der_bound_initial_value = sum(Δapv .* bd_iv_v)
    bv_fv_v = PatternVector(n, FinalValuePattern(zero(T), one(T)))
    der_bound_final_value = sum(Δapv .* bv_fv_v)
    val_der = sum(Δapv) - der_bound_initial_value - der_bound_final_value
    return PaddedFillPattern(der_bound_initial_value, val_der, der_bound_final_value)
end

#Since we use a sum function in the pullback, we implement it for PaddedFillPattern
function Base.sum(x::PatternVector{T, P}) where {T, P <: PaddedFillPattern{T}}
    pattern = x.pattern
    return muladd(x.n - 2, pattern.value, pattern.bound_initial_value + pattern.bound_final_value)
end
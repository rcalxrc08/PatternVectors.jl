
## FinalValuePattern
# FinalValuePattern represents a pattern that has all of the values that are the same apart from the last.
struct FinalValuePattern{T} <: AbstractPattern{T}
    other_value::T
    final_value::T
    function FinalValuePattern(other_value::T, final_value::T) where {T}
        return new{T}(other_value, final_value)
    end
end

pattern_minimum_size(x::FinalValuePattern) = 2 # COV_EXCL_LINE

function getindex_pattern(x::FinalValuePattern, ind::Int, n::Int)
    ifelse(ind == n, x.final_value, x.other_value)
end

function getindex_pattern_range(x::P, el::AbstractRange{T}, n::Int) where {T <: Int, P <: FinalValuePattern}
    new_len = length(el)
    minimum_size = pattern_minimum_size(x)
    (minimum_size <= new_len) || throw(DomainError(new_len, "Trying to getindex with an AbstractRange of length $new_len. Provided length must be greater or equal to $minimum_size."))
    first_idx = el.start
    iv_value = getindex_pattern(x, first_idx, n)
    final_value = getindex_pattern(x, last(el), n)
    return new_len, FinalValuePattern(iv_value, final_value)
end

function materialize_pattern(bc::Base.Broadcast.Broadcasted{ArrayStylePatternVector{P}, Nothing, <:F, <:R}) where {F, R, P <: FinalValuePattern}
    bc_f = Broadcast.flatten(bc)
    func = bc_f.f
    args = bc_f.args
    axes_result = Broadcast.combine_axes(args...)
    n = length(first(axes_result))
    iv_part = func(flat_index.(args, 1)...)
    end_part = func(flat_index.(args, n)...)
    return length(first(axes_result)), FinalValuePattern(iv_part, end_part)
end

determine_mixed_pattern(::Type{T}, ::Type{V}) where {T <: FinalValuePattern{L}, V <: FinalValuePattern{N}} where {L, N} = FinalValuePattern{promote_type(L, N)}

# Function to determine the mixed pattern type when combining various patterns
determine_mixed_pattern(::Type{T}, ::Type{V}) where {T <: ZeroPattern{L}, V <: FinalValuePattern{N}} where {L, N} = FinalValuePattern{promote_type(L, N)}
determine_mixed_pattern(::Type{T}, ::Type{V}) where {T <: FillPattern{L}, V <: FinalValuePattern{N}} where {L, N} = FinalValuePattern{promote_type(L, N)}

function ChainRulesCore.rrule(::Type{FinalValuePattern}, args...)
    function AbstractPattern_pb(Δapv)
        NoTangent(), (getfield(Δapv, arg) for arg in fieldnames(FinalValuePattern))...
    end
    return FinalValuePattern(args...), AbstractPattern_pb
end

function pattern_to_vector_pullback(::Type{P}, Δapv, n) where {P <: FinalValuePattern{T}} where {T}
    bd_val_v = PatternVector(n, FinalValuePattern(one(T), zero(T)))
    der_bound_initial_value = sum(Δapv .* bd_val_v)
    der_bound_final_value = sum(Δapv) - der_bound_initial_value
    return FinalValuePattern(der_bound_initial_value, der_bound_final_value)
end

#Since we use a sum function in the pullback, we implement it for FinalValuePattern
function Base.sum(x::PatternVector{T, P}) where {T, P <: FinalValuePattern{T}}
    pattern = x.pattern
    return muladd(x.n - 1, pattern.other_value, pattern.final_value)
end
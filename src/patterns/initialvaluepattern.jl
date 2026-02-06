
## InitialValuePattern
# InitialValuePattern represents a pattern that has all of the values that are the same apart from the first.
struct InitialValuePattern{T} <: AbstractPattern{T}
    first_value::T
    other_value::T
    function InitialValuePattern(first_value::T, other_value::T) where {T}
        return new{T}(first_value, other_value)
    end
end

pattern_minimum_size(::InitialValuePattern) = 2 # COV_EXCL_LINE

function getindex_pattern(x::InitialValuePattern, ind::Int, ::Int)
    ifelse(ind == 1, x.first_value, x.other_value)
end

function getindex_pattern_range(x::P, el::AbstractRange{T}, n::Int) where {T <: Int, P <: InitialValuePattern}
    new_len = length(el)
    minimum_size = pattern_minimum_size(x)
    (minimum_size <= new_len) || throw(DomainError(new_len, "Trying to getindex with an AbstractRange of length $new_len. Provided length must be greater or equal to $minimum_size."))
    first_idx = el.start
    @views @inbounds iv_value = getindex_pattern(x, first_idx, n)
    @views @inbounds ov_value = getindex_pattern(x, first_idx + step(el), n)
    return new_len, InitialValuePattern(iv_value, ov_value)
end

function materialize_pattern(bc::Base.Broadcast.Broadcasted{ArrayStylePatternVector{P}, Nothing, <:F, <:R}) where {F, R, P <: InitialValuePattern}
    bc_f = Broadcast.flatten(bc)
    func = bc_f.f
    args = bc_f.args
    axes_result = Broadcast.combine_axes(args...)
    iv_part = func(flat_index.(args, 1)...)
    ov_part = func(flat_index.(args, 2)...)
    return length(first(axes_result)), InitialValuePattern(iv_part, ov_part)
end

# Function to determine the mixed pattern type when combining various patterns
determine_mixed_pattern(::Type{T}, ::Type{V}) where {T <: ZeroPattern{L}, V <: InitialValuePattern{N}} where {L, N} = InitialValuePattern{promote_type(L, N)}
determine_mixed_pattern(::Type{T}, ::Type{V}) where {T <: FillPattern{L}, V <: InitialValuePattern{N}} where {L, N} = InitialValuePattern{promote_type(L, N)}
determine_mixed_pattern(::Type{T}, ::Type{V}) where {T <: InitialValuePattern{L}, V <: InitialValuePattern{N}} where {L, N} = InitialValuePattern{promote_type(L, N)}

#Mixture generation
determine_mixed_pattern(::Type{T}, ::Type{V}) where {T <: EvenOddPattern{L}, V <: InitialValuePattern{N}} where {L, N} = PaddedEvenOddPattern{promote_type(L, N)}

function ChainRulesCore.rrule(::Type{InitialValuePattern}, args...)
    function AbstractPattern_pb(Δapv)
        NoTangent(), (getfield(Δapv, arg) for arg in fieldnames(InitialValuePattern))...
    end
    return InitialValuePattern(args...), AbstractPattern_pb
end

function pattern_to_vector_pullback(::Type{P}, Δapv, n) where {P <: InitialValuePattern{T}} where {T}
    bd_val_v = PatternVector(n, InitialValuePattern(one(T), zero(T)))
    der_bound_initial_value = sum(Δapv .* bd_val_v)
    der_bound_final_value = sum(Δapv) - der_bound_initial_value
    return InitialValuePattern(der_bound_initial_value, der_bound_final_value)
end

#Since we use a sum function in the pullback, we implement it for InitialValuePattern
function Base.sum(x::PatternVector{T, P}) where {T, P <: InitialValuePattern{T}}
    pattern = x.pattern
    return muladd(x.n - 1, pattern.other_value, pattern.first_value)
end
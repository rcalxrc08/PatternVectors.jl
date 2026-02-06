## FillPattern
# FillPattern represents a pattern that fills all indices with the same value.
struct FillPattern{T} <: AbstractPattern{T}
    value::T
    function FillPattern(x::T) where {T}
        return new{T}(x)
    end
end

pattern_minimum_size(::P) where {P <: FillPattern} = 1

function getindex_pattern(x::FillPattern, _::Int, ::Int)
    x.value
end
function getindex_pattern_range(x::FillPattern, el::AbstractRange{T}, n::Int) where {T <: Int}
    new_len = length(el)
    @views @inbounds bound_final_value = getindex_pattern(x, last(el), n)
    return new_len, FillPattern(bound_final_value)
end

function materialize_pattern(bc::Base.Broadcast.Broadcasted{ArrayStylePatternVector{P}, Nothing, <:F, <:R}) where {F, R, P <: FillPattern}
    bc_f = Broadcast.flatten(bc)
    func = bc_f.f
    args = bc_f.args
    axes_result = Broadcast.combine_axes(args...)
    value = func(flat_index.(args, 1)...)
    return length(first(axes_result)), FillPattern(value)
end

# Function to determine the mixed pattern type when combining various patterns
determine_mixed_pattern(::Type{T}, ::Type{V}) where {T <: ZeroPattern{L}, V <: FillPattern{M}} where {L, M} = FillPattern{promote_type(L, M)}

function ChainRulesCore.rrule(::Type{FillPattern}, args...)
    function AbstractPattern_pb(Δapv)
        NoTangent(), (getfield(Δapv, arg) for arg in fieldnames(FillPattern))...
    end
    return FillPattern(args...), AbstractPattern_pb
end

function pattern_to_vector_pullback(::Type{P}, Δapv, n) where {P <: FillPattern{T}} where {T}
    return FillPattern(sum(Δapv))
end

#Since we use a sum function in the pullback, we implement it for FillPattern
function Base.sum(x::PatternVector{T, P}) where {T, P <: FillPattern{T}}
    return x.n * x.pattern.value
end
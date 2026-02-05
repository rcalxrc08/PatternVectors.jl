## ZeroPattern

# ZeroPattern represents a pattern that fills all indices with zeros.

struct ZeroPattern{T} <: AbstractPattern{T}
    function ZeroPattern(::T) where {T}
        return new{T}()
    end
end

pattern_minimum_size(::Type{P}) where {P <: ZeroPattern} = 1

function getindex_pattern(::ZeroPattern{T}, ::Int, ::Int) where {T}
    zero(T)
end

function getindex_pattern_range(::ZeroPattern{V}, el::AbstractRange{T}, ::Int) where {T <: Int, V}
    new_len = length(el)
    return new_len, ZeroPattern(zero(V))
end

function materialize_pattern(bc::Base.Broadcast.Broadcasted{ArrayStylePatternVector{P}, Nothing, <:F, <:R}) where {F, R, P <: ZeroPattern}
    bc_f = Broadcast.flatten(bc)
    func = bc_f.f
    args = bc_f.args
    axes_result = Broadcast.combine_axes(args...)
    value = func(flat_index.(args, 1)...)
    return length(first(axes_result)), FillPattern(value)
end

# ZeroPattern will generate a FillPattern upon materialization
function Base.materialize(bc::Base.Broadcast.Broadcasted{ArrayStylePatternVector{P}, Nothing, <:F, <:R}) where {F, R, P <: ZeroPattern{T}} where {T}
    length_first_axes_result, pattern = materialize_pattern(bc)
    return PatternVector(length_first_axes_result, pattern)
end

function ChainRulesCore.rrule(::Type{V}, args...) where {V <: ZeroPattern{T}} where {T}
    function AbstractPattern_pb(_)
        NoTangent(), NoTangent()
    end
    return ZeroPattern(args...), AbstractPattern_pb
end

function pattern_to_vector_pullback(::Type{P}, Δapv, n) where {P <: ZeroPattern{T}} where {T}
    return ZeroPattern(zero(T))
end
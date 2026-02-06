#Broacasting over PatternVector
abstract type AbstractArrayStylePatternVector <: Broadcast.AbstractArrayStyle{1} end
# Pattern is stored inside the style, to allow easier mixture patterns dispatching
struct ArrayStylePatternVector{P <: AbstractPattern} <: AbstractArrayStylePatternVector end
# Main link between PatternVector and its Broadcast style, unfortunately we are carrying the element type too.
Base.BroadcastStyle(::Type{<:PatternVector{T, P}}) where {T, P <: AbstractPattern} = ArrayStylePatternVector{P}()
# Mixture pattern determination for same pattern
# determine_mixed_pattern(::Type{T}, ::Type{T}) where {T <: AbstractPattern} = T
# Mixture pattern determination for pattern with Unknown
determine_mixed_pattern(::Type{T}, ::Type{UnknownPattern}) where {T <: AbstractPattern} = T
determine_mixed_pattern(::Type{UnknownPattern}, ::Type{T}) where {T <: AbstractPattern} = T
# Default mixture pattern determination returns Unknown
determine_mixed_pattern(::Type{T}, ::Type{V}) where {T <: AbstractPattern, V <: AbstractPattern} = UnknownPattern

# Defer mixture pattern determination to handle symmetric cases
function determine_mixed_pattern_defer(::Type{T}, ::Type{V}) where {T <: AbstractPattern, V <: AbstractPattern}
    first = determine_mixed_pattern(T, V)
    second = determine_mixed_pattern(V, T)
    return determine_mixed_pattern(first, second)
end

# Mixture pattern determination for two ArrayStylePatternVector
Base.BroadcastStyle(::ArrayStylePatternVector{T}, ::ArrayStylePatternVector{V}) where {T <: AbstractPattern, V <: AbstractPattern} = ArrayStylePatternVector{determine_mixed_pattern_defer(T, V)}()

# Broacasting over PatternVector - flattening helpers
flat_index(x, _) = x
flat_index(x::AbstractArray{T, 0}, _) where {T} = x[]
flat_index(x::Base.RefValue, _) = x.x
flat_index(x::PatternVector, ind) = @views @inbounds x[ind]

# Dispatch materialization based on pattern type
function Base.materialize(bc::Base.Broadcast.Broadcasted{ArrayStylePatternVector{P}, Nothing, <:F, <:R}) where {F, R, P <: AbstractPattern{T}} where {T}
    length_first_axes_result, pattern = materialize_pattern(bc)
    return PatternVector(length_first_axes_result, pattern)
end

# Defines the broadcast style for ArrayStylePatternVector and a Tuple broadcast style.
Base.BroadcastStyle(::ArrayStylePatternVector, ::Base.Broadcast.Style{Tuple}) = Broadcast.DefaultArrayStyle{1}()

# Determines the broadcast style when combining ArrayStylePatternVector with a DefaultArrayStyle of any dimension.
function Base.BroadcastStyle(a::ArrayStylePatternVector, ::Broadcast.DefaultArrayStyle{N}) where {N}
    if (N > 0)
        return PatternMixtureArrayStyle{N, Broadcast.DefaultArrayStyle{N}}()
    end
    return a
end

# Determines the broadcast style when combining ArrayStylePatternVector with a general AbstractArrayStyle of any dimension.
function Base.BroadcastStyle(a::ArrayStylePatternVector, ::T) where {T <: Broadcast.AbstractArrayStyle{N}} where {N}
    if (N > 0)
        return PatternMixtureArrayStyle{N, T}()
    end
    return Base.BroadcastStyle(a, Broadcast.DefaultArrayStyle{0}())
end

# Materializes a broadcasted object if its style is ArrayStylePatternVector.
function materialize_if_needed(bc::Base.Broadcast.Broadcasted{T, Nothing, <:F, <:R}) where {T <: ArrayStylePatternVector, F, R}
    return Base.materialize(bc)
end

# Exclude Unknown broadcast style, using the second style as the fallback.
function exclude_unknown_style_if_possible(::Base.Broadcast.Unknown, style_1::V, style_2::U) where {V <: Base.Broadcast.BroadcastStyle, U <: Base.Broadcast.BroadcastStyle}
    Base.BroadcastStyle(style_2, style_1)
end

# Prefer the first style if it is not Unknown.
function exclude_unknown_style_if_possible(first::L, ::V, ::U) where {L <: Base.Broadcast.BroadcastStyle, V <: Base.Broadcast.BroadcastStyle, U <: Base.Broadcast.BroadcastStyle}
    first
end

# Optimization for broadcasting

# Represents a broadcast style that mixes two other broadcast styles.
abstract type AbstractPatternMixtureArrayStyle{N} <: Broadcast.AbstractArrayStyle{N} end

struct PatternMixtureArrayStyle{N, T} <: AbstractPatternMixtureArrayStyle{N}
    # Construct from a single broadcast style.
    function PatternMixtureArrayStyle{N, V}() where {V <: Broadcast.AbstractArrayStyle{N}} where {N}
        return new{N, V}()
    end
    function PatternMixtureArrayStyle{V}() where {V <: Broadcast.AbstractArrayStyle{N}} where {N}
        return new{N, V}()
    end
    # Construct by mixing two broadcast styles, resolving Unknown if possible.
    function PatternMixtureArrayStyle(style_1::V, style_2::U) where {V <: Base.Broadcast.BroadcastStyle, U <: Base.Broadcast.BroadcastStyle}
        style_implied_1 = Base.BroadcastStyle(style_1, style_2)
        style = exclude_unknown_style_if_possible(style_implied_1, style_1, style_2)
        return PatternMixtureArrayStyle{typeof(style)}()
    end
end

function get_style(::PatternMixtureArrayStyle{N, T}) where {T, N}
    return T()
end

# BroadcastStyle rules for PatternMixtureArrayStyle

# Combine PatternMixtureArrayStyle with a general AbstractArrayStyle.
function Base.BroadcastStyle(a::PatternMixtureArrayStyle, b::Broadcast.AbstractArrayStyle{N}) where {N}
    return PatternMixtureArrayStyle(get_style(a), b)
end

# Combine PatternMixtureArrayStyle with a DefaultArrayStyle.
function Base.BroadcastStyle(a::PatternMixtureArrayStyle, b::Broadcast.DefaultArrayStyle{N}) where {N}
    return PatternMixtureArrayStyle(get_style(a), b)
end

# Combine two PatternMixtureArrayStyle objects by mixing their sub-styles.
function Base.BroadcastStyle(a::PatternMixtureArrayStyle, b::PatternMixtureArrayStyle)
    return PatternMixtureArrayStyle(get_style(a), get_style(b))
end

# Materialize broadcasted objects with PatternMixtureArrayStyle.
function materialize_if_needed(bc::Base.Broadcast.Broadcasted{PatternMixtureArrayStyle{N, T}, Nothing, <:F, <:R}) where {N, T, F, R}
    return Base.materialize(bc)
end

# Default: return object unchanged if materialization is not needed.
function materialize_if_needed(bc)
    return bc
end

# Materialize a broadcasted object with PatternMixtureArrayStyle, materializing all arguments as needed.
function Base.materialize(bc::Base.Broadcast.Broadcasted{PatternMixtureArrayStyle{N, T}, Nothing, <:F, <:R}) where {N, T, F, R}
    mat_args = materialize_if_needed.(bc.args)
    res = Base.materialize(Base.Broadcast.Broadcasted(get_style(bc.style), bc.f, mat_args))
    return res
end
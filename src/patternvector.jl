abstract type AbstractPatternUntyped end

abstract type AbstractPattern{T} <: AbstractPatternUntyped end

struct UnknownPattern <: AbstractPatternUntyped end

#Implementation of the concrete class PatternVector
struct PatternVector{T, P} <: AbstractArray{T, 1}
    n::Int64
    pattern::P
    function PatternVector(n::Int64, pattern::P) where {P <: AbstractPattern{T}} where {T}
        minimum_size = pattern_minimum_size_defer(P)
        (minimum_size <= n) || throw(DomainError(n, "length of PatternVector for pattern $P must be greater or equal to $minimum_size. Provided is $n."))
        return new{T, P}(n, pattern)
    end
end

# Customizes how PatternVector types are shown in argument lists (REPL, error messages).
Base.showarg(io::IO, A::PatternVector, _) = print(io, typeof(A))

### Implementation of the array interface

# Returns the size (length) of the PatternVector.
Base.size(A::PatternVector) = (A.n,)

# Allows colon indexing to return the vector itself.
Base.getindex(x::PatternVector, ::Colon) = x

function pattern_minimum_size_defer(::Type{P}) where {P <: AbstractPattern}
    return pattern_minimum_size(P)
end

# Defer get_index to pattern
function Base.getindex(x::PatternVector, ind::Int)
    @boundscheck (1 <= ind <= x.n) || throw(BoundsError(x, ind))
    getindex_pattern(x.pattern, ind, x.n)
end

# Defer get_index to pattern
function Base.getindex(x::PatternVector{V, P}, el::AbstractRange{T}) where {T <: Int, V, P}
    @boundscheck (1 <= el.start <= x.n) || throw(BoundsError(x, el.start))
    @boundscheck (1 <= last(el) <= x.n) || throw(BoundsError(x, last(el)))
    n, pattern = getindex_pattern_range(x.pattern, el, x.n)
    return PatternVector(n, pattern)
end

abstract type AbstractPatternUntyped end

abstract type AbstractPattern{T} <: AbstractPatternUntyped end

struct UnknownPattern <: AbstractPatternUntyped end

"""
    PatternVector(n, pattern)

Immutable `AbstractVector` of length `n` whose elements are given by `pattern`, an `AbstractPattern`: only `n` and the
values of the pattern are stored, so a `PatternVector` of `isbits` values is `isbits`. All the values of the pattern
have the same type, the element type, and `n` must be at least the minimum length of the pattern.

A broadcast on `PatternVector`s and scalars only evaluates the function once per distinct value and returns a
`PatternVector`; mixed with another array it returns an array of the type of the other one. Indexing with a range
returns a `PatternVector`, and `sum` is `O(1)`.

```jldoctest
julia> x = PatternVector(6, EvenOddPattern(1.0, 2.0))
6-element PatternVector{Float64, EvenOddPattern{Float64}}:
 1.0
 2.0
 1.0
 2.0
 1.0
 2.0

julia> sum(x)
9.0
```
"""
struct PatternVector{T, P} <: AbstractArray{T, 1}
    n::Int64
    pattern::P
    function PatternVector(n::Int64, pattern::P) where {P <: AbstractPattern{T}} where {T}
        minimum_size = pattern_minimum_size(pattern)
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

# Keyword calls such as `sum(x; dims = :)` (used by Zygote's broadcast pullback) dispatch only among
# methods accepting keywords, so they would skip the pattern-specialized `sum` methods and iterate
# over all n elements. Forward them to the positional `sum`, leaving positional dispatch untouched.
function Core.kwcall(kws::NamedTuple, ::typeof(sum), x::PatternVector)
    kws === (dims = :,) && return sum(x)
    return invoke(Core.kwcall, Tuple{NamedTuple, typeof(sum), AbstractArray}, kws, sum, x)
end

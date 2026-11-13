```@raw html
<div class="fc-page-head">
<p class="fc-kicker"><span>Guide 04</span> Define your own pattern</p>
```

# New patterns

A pattern is an immutable struct subtyping `AbstractPattern{T}`, and it can be defined outside the package. The
methods below are all that is needed; as an example, a pattern repeating three values, `[a, b, c, a, b, c, …]`:

```@raw html
</div>
```

```@example cyclic
using PatternVectors, ChainRulesCore
import PatternVectors: AbstractPattern, ArrayStylePatternVector, pattern_minimum_size, getindex_pattern,
    getindex_pattern_range, materialize_pattern, flat_index, determine_mixed_pattern, pattern_to_vector_pullback

struct CyclicPattern{T} <: AbstractPattern{T}
    a::T
    b::T
    c::T
end

# 1. Smallest length the pattern is defined for
pattern_minimum_size(::CyclicPattern) = 3

# 2. Element at index `ind` of a vector of length `n`
getindex_pattern(x::CyclicPattern, ind::Int, ::Int) = (x.a, x.b, x.c)[mod1(ind, 3)]

# 3. Indexing with a range: returns the new length and pattern
getindex_pattern_range(x::CyclicPattern, r::AbstractRange{Int}, n::Int) =
    length(r), CyclicPattern(ntuple(k -> getindex_pattern(x, r[k], n), 3)...)

# 4. Broadcast: evaluate the function once per distinct value (`flat_index(arg, k)` is element k of any argument)
function materialize_pattern(bc::Base.Broadcast.Broadcasted{ArrayStylePatternVector{P}}) where {P <: CyclicPattern}
    bc_f = Broadcast.flatten(bc)
    n = length(first(Broadcast.combine_axes(bc_f.args...)))
    return n, CyclicPattern(ntuple(k -> bc_f.f(flat_index.(bc_f.args, k)...), 3)...)
end

# 5. Pattern resulting from broadcasting with other patterns (UnknownPattern, i.e. no result, otherwise)
determine_mixed_pattern(::Type{<:CyclicPattern{L}}, ::Type{<:CyclicPattern{M}}) where {L, M} = CyclicPattern{promote_type(L, M)}
determine_mixed_pattern(::Type{<:FillPattern{L}}, ::Type{<:CyclicPattern{M}}) where {L, M} = CyclicPattern{promote_type(L, M)}
determine_mixed_pattern(::Type{<:ZeroPattern{L}}, ::Type{<:CyclicPattern{M}}) where {L, M} = CyclicPattern{promote_type(L, M)}

# 6. Optional but recommended: O(1) sum (the default iterates over all elements)
function Base.sum(x::PatternVector{T, <:CyclicPattern}) where {T}
    q, r = divrem(x.n, 3)
    return (q + (r >= 1)) * x.pattern.a + (q + (r >= 2)) * x.pattern.b + q * x.pattern.c
end
nothing # hide
```

`determine_mixed_pattern` is checked in both argument orders, so each pair is defined once. The new pattern works as
the others:

```@example cyclic
c = PatternVector(8, CyclicPattern(1.0, 2.0, 3.0))
exp.(c) .+ PatternVector(8, FillPattern(1.0))
```

## Differentiating a new pattern

Enzyme and Mooncake need nothing more. Mooncake's rule for broadcasts mixing `PatternVector`s and arrays
differentiates `getindex_pattern` on the pattern rebuilt from dual numbers with its positional constructor
(`CyclicPattern(a, b, c)`), so all fields must have the element type. Zygote needs `pattern_to_vector_pullback`, which
maps a cotangent vector `Δ` (dense or `PatternVector`) onto the pattern fields, together with the rule for the
constructor:

```julia
function ChainRulesCore.rrule(::Type{CyclicPattern}, a, b, c)
    CyclicPattern_pb(Δ) = NoTangent(), Δ.a, Δ.b, Δ.c
    return CyclicPattern(a, b, c), CyclicPattern_pb
end

function pattern_to_vector_pullback(::Type{<:CyclicPattern{T}}, Δ, n) where {T}
    o, z = one(T), zero(T)
    return CyclicPattern(sum(Δ .* PatternVector(n, CyclicPattern(o, z, z))),
                         sum(Δ .* PatternVector(n, CyclicPattern(z, o, z))),
                         sum(Δ .* PatternVector(n, CyclicPattern(z, z, o))))
end
```

The patterns in [`src/patterns`](https://github.com/rcalxrc08/PatternVectors.jl/tree/main/src/patterns) follow the
same structure.

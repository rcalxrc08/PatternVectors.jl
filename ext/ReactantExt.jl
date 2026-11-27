module ReactantExt

using PatternVectors
using PatternVectors: getindex_pattern, AbstractPattern
using Reactant
using Reactant: TracedRNumber, TracedRArray, Ops

# Reactant (XLA) never receives the n elements of a PatternVector. Where XLA needs the vector as an array (a broadcast
# with a traced array, an operation of Reactant on it), the array is built from the pattern values with the structural
# operations of XLA: a fill (broadcast of a scalar), the concatenation of the bounds, and an alternating pair repeated
# by a broadcast and a reshape. The program holds only the pattern values, constant or traced; neither Reactant nor
# XLA folds a broadcast into a constant, and XLA fuses these operations into the operation that consumes them.
# Without this Reactant converts the PatternVector into a dense constant of n values embedded in the program.
# Operations on PatternVectors only (broadcasts with scalars, sum, dot with a FillPattern) never get here: they work
# on the pattern values, traced numbers included, and return a PatternVector or a scalar.

# A pattern value as a traced scalar. A number becomes a constant behind an optimization barrier: otherwise Reactant
# (below 1024 elements) and XLA (a concatenation, the comparisons with an iota) fold the array built from it into a
# dense constant of n values. The barrier is on the scalars only: the array built from them still fuses.
# T is the element type of the pattern, a traced number when the pattern values are traced.
traced_value(::Type{T}, v::TracedRNumber) where {T} = Reactant.promote_to(TracedRNumber{Reactant.unwrapped_eltype(T)}, v)
traced_value(::Type{T}, v) where {T} = only(Ops.optimization_barrier(Reactant.promote_to(TracedRNumber{Reactant.unwrapped_eltype(T)}, v)))
# m copies of a value
filled(::Type{T}, v, m) where {T} = Reactant.broadcast_to_size(traced_value(T, v), (m,))
# a, b, a, b, ... of length m
function alternating(::Type{T}, a, b, m) where {T}
    pair = vcat(filled(T, a, 1), filled(T, b, 1))
    half = cld(m, 2)
    repeated = Ops.reshape(Ops.broadcast_in_dim(pair, [1], [2, half]), [2 * half])
    return m == 2 * half ? repeated : Ops.slice(repeated, [1], [m])
end
# concatenation along the only dimension, skipping empty parts
joined(parts...) = Ops.concatenate([p for p in parts if length(p) > 0], 1)

# a fill is a broadcast of a scalar, which is never folded: no barrier, so that XLA still simplifies with the value
broadcast_value(::Type{T}, v, n) where {T} = Reactant.broadcast_to_size(Reactant.promote_to(TracedRNumber{Reactant.unwrapped_eltype(T)}, v), (n,))
traced_array(p::FillPattern{T}, n) where {T} = broadcast_value(T, p.value, n)
traced_array(::ZeroPattern{T}, n) where {T} = broadcast_value(T, zero(Reactant.unwrapped_eltype(T)), n)
traced_array(p::EvenOddPattern{T}, n) where {T} = alternating(T, p.value_odd, p.value_even, n)
traced_array(p::InitialValuePattern{T}, n) where {T} = joined(filled(T, p.first_value, 1), filled(T, p.other_value, n - 1))
traced_array(p::FinalValuePattern{T}, n) where {T} = joined(filled(T, p.other_value, n - 1), filled(T, p.final_value, 1))
traced_array(p::PaddedFillPattern{T}, n) where {T} = joined(filled(T, p.bound_initial_value, 1), filled(T, p.value, n - 2), filled(T, p.bound_final_value, 1))
function traced_array(p::PaddedEvenOddPattern{T}, n) where {T}
    inner = alternating(T, p.value_even, p.value_odd, n - 2) # n ≥ 4
    return joined(filled(T, p.bound_initial_value, 1), inner, filled(T, p.bound_final_value, 1))
end
# Any other pattern: its getindex_pattern on an iota, compares and selects on the index. Reactant folds them into a
# constant for n ≤ 1024 and XLA folds the masks of the comparisons with the index: prefer a method above.
function traced_array(p::AbstractPattern, n)
    index = Ops.iota(Int64, [n]; iota_dimension = 1) .+ 1
    return broadcast(i -> getindex_pattern(p, i, n), index)
end

traced_array(x::PatternVector) = traced_array(x.pattern, length(x))

# A mixed broadcast: Reactant brings every argument to the size of the result with broadcast_to_size.
Reactant.broadcast_to_size(x::PatternVector, rsize) = Reactant.broadcast_to_size(traced_array(x), rsize)

# Operations of Reactant that need a traced array (sorting, cumulative sums, linear algebra, ...).
Reactant.ReactantCore.materialize_traced_array(x::PatternVector) = traced_array(x)
# Reactant's indexing of an array of traced numbers looks for the traced array it wraps: a PatternVector wraps none.
Reactant.TracedUtils.get_ancestor_and_indices(x::PatternVector, indices...) = (traced_array(x), indices)
Reactant.TracedUtils.get_ancestor_and_indices(x::PatternVector, indices, args...) = (traced_array(x), (indices, args...))

# A PatternVector of traced numbers is an AbstractArray{<:TracedRNumber}, for which Reactant defines indexing too.
const TracedPatternVector = PatternVector{<:TracedRNumber}
# A traced index: the pattern at that index, compares and selects on the pattern values (no bounds check: the index
# has no value while tracing).
Base.getindex(x::PatternVector, ind::TracedRNumber{<:Integer}) = getindex_pattern(x.pattern, ind, length(x))
Base.getindex(x::TracedPatternVector, ind::TracedRNumber{Int}) = getindex_pattern(x.pattern, ind, length(x))
function Base.getindex(x::TracedPatternVector, ind::Int)
    @boundscheck (1 <= ind <= length(x)) || throw(BoundsError(x, ind))
    return getindex_pattern(x.pattern, ind, length(x))
end
# The output of the operations that fill an array (cumsum, collect, map, ...) is a traced array.
Base.similar(x::TracedPatternVector, ::Type{T}, dims::Dims) where {T} = similar(traced_array(x), T, dims)
Base.collect(x::TracedPatternVector) = traced_array(x)

end

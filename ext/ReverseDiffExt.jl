module ReverseDiffExt

using ReverseDiff, PatternVectors, LinearAlgebra

# Since ReverseDiff 1.18 TrackedStyle is an AbstractArrayStyle, and both packages claim precedence over every
# AbstractArrayStyle: the two argument orders disagree and Base reports conflicting broadcast rules.
# The tracked style must win, so both orders defer to the ReverseDiff rule: ReverseDiff then broadcasts over the
# values, where the PatternVectors take their own path.
Base.BroadcastStyle(a::PatternVectors.ArrayStylePatternVector, s::ReverseDiff.TrackedStyle) = Base.BroadcastStyle(s, a)
Base.BroadcastStyle(a::PatternVectors.PatternMixtureArrayStyle, s::ReverseDiff.TrackedStyle) = Base.BroadcastStyle(s, a)

# ReverseDiff records dot(::AbstractVector, ::TrackedArray) and the opposite order: the dot of PatternVectors is as
# specific in the other argument, so the pairs with a tracked array are defined here, recorded as ReverseDiff does.
LinearAlgebra.dot(x::PatternVector, y::ReverseDiff.TrackedArray{Y, D}) where {Y, D} = ReverseDiff.record_dot(x, y, D)
LinearAlgebra.dot(x::ReverseDiff.TrackedArray{X, D}, y::PatternVector) where {X, D} = ReverseDiff.record_dot(x, y, D)

end # module

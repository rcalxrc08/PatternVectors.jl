module ReverseDiffExt

using ReverseDiff, PatternVectors

# Since ReverseDiff 1.18 TrackedStyle is an AbstractArrayStyle, and both packages claim precedence over every
# AbstractArrayStyle: the two argument orders disagree and Base reports conflicting broadcast rules.
# The tracked style must win, so both orders defer to the ReverseDiff rule: ReverseDiff then broadcasts over the
# values, where the PatternVectors take their own path.
Base.BroadcastStyle(a::PatternVectors.ArrayStylePatternVector, s::ReverseDiff.TrackedStyle) = Base.BroadcastStyle(s, a)
Base.BroadcastStyle(a::PatternVectors.PatternMixtureArrayStyle, s::ReverseDiff.TrackedStyle) = Base.BroadcastStyle(s, a)

end # module

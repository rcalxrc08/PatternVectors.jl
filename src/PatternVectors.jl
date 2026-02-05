module PatternVectors

include("patternvector.jl")
include("broadcast.jl")
include("rrules.jl")
include("zeropattern.jl")
include("fillpattern.jl")
include("evenoddpattern.jl")
include("paddedevenoddpattern.jl")

export PatternVector, ZeroPattern, FillPattern, EvenOddPattern, PaddedEvenOddPattern

end

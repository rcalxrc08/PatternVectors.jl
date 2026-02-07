module PatternVectors

include("patternvector.jl")
include("broadcast.jl")
include("rrules.jl")
include("patterns/zeropattern.jl")
include("patterns/fillpattern.jl")
include("patterns/evenoddpattern.jl")
include("patterns/initialvaluepattern.jl")
include("patterns/finalvaluepattern.jl")
include("patterns/paddedfillpattern.jl")
include("patterns/paddedevenoddpattern.jl")

export PatternVector, ZeroPattern, FillPattern, EvenOddPattern, PaddedEvenOddPattern, InitialValuePattern, FinalValuePattern, PaddedFillPattern

end

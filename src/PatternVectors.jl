module PatternVectors
using ChainRulesCore
include("01_vector/patternvector.jl")
include("02_broadcast/broadcast.jl")
include("03_rrules/rrules.jl")
include("04_patterns/zeropattern.jl")
include("04_patterns/fillpattern.jl")
include("04_patterns/evenoddpattern.jl")
include("04_patterns/initialvaluepattern.jl")
include("04_patterns/finalvaluepattern.jl")
include("04_patterns/paddedfillpattern.jl")
include("04_patterns/paddedevenoddpattern.jl")
include("05_linearalgebra/linearalgebra.jl")

export PatternVector, ZeroPattern, FillPattern, EvenOddPattern, PaddedEvenOddPattern, InitialValuePattern, FinalValuePattern, PaddedFillPattern

end

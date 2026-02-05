module SparseArraysExt

using SparseArrays, PatternVectors

Base.BroadcastStyle(::PatternVectors.ArrayStylePatternVector, ::SparseArrays.HigherOrderFns.SparseVecStyle) = SparseArrays.HigherOrderFns.PromoteToSparse()
SparseArrays.HigherOrderFns.is_supported_sparse_broadcast(::PatternVectors.PatternVector, rest...) = SparseArrays.HigherOrderFns.is_supported_sparse_broadcast(rest...)

end # module
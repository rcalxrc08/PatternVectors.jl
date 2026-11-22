# Testing

_What each test file checks, reference values, local versus CI._ (Layer 3 reference.)

| Test file | What | Reference values from |
|:--|:--|:--|
| `test/runtests.jl` | every pattern (indexing, ranges, broadcasts, `sum`), "Nested mixed broadcasts" (memory included), `dot`, sparse | the dense `collect` of the same vector |
| `test/test_Zygote.jl` | gradients per pattern, ChainRules pullbacks called directly, keyword `sum`, deferred cotangents, `dot` | Zygote on the dense vector |
| `test/test_Mooncake.jl` | structural tangent, the mixed rule (non-default style, Int8 weights, `ZeroPattern`), allocations per array | Zygote |
| `test/test_ReverseDiff.jl` | style precedence, gradients, `dot` | Zygote / dense |
| `test/test_Enzyme.jl` | scalar gradients, pattern gradients, size independence (n = 10^12) | Zygote, Mooncake |
| `test/cuda_tests.jl` | Zygote and Mooncake on `CuArray`s (opt-in, needs a GPU) | the CPU result |
| `test/context.jl` | the context layers follow the code | — |

- Line coverage is 100% with the CI settings (`--inline=yes`).
- `cuda_tests.jl` runs in an environment with PatternVectors (dev), CUDA, Zygote, ChainRulesCore, BenchmarkTools,
  Mooncake, Statistics.
- Mooncake's `TestUtils.test_rule` (reverse mode, `unsafe_perturb = true`) passes on the rule for the tested cases
  (checked by hand, not in the suite).
- Locally on Julia 1.13, `test_Enzyme.jl` aborted at line 37 even on Enzyme 0.13.218 (`handleKnownCallDerivatives`), not
  on CI (1.11, 1.12).
- Allocations: `@allocated` on a function called once before, inside a function.

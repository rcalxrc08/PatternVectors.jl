# Dependencies

_Why each dependency exists, hard or weak, and its compat floor._ (Layer 3 reference.)

| Package | Hard/weak | Why | Compat floor and why |
|:--|:--|:--|:--|
| `ChainRulesCore` | hard | the Zygote rules (stages 03, 04, 05) | 1 |
| `LinearAlgebra` | hard | `dot` | 1 |
| `Mooncake` | weak (`MooncakeExt`) | the mixed-broadcast rule | `0.5`; the extension uses `Mooncake.Nfwd`, which exists since 0.5.25: the floor should be raised (`state/STATUS.md`) |
| `ReverseDiff` | weak (`ReverseDiffExt`) | broadcast style precedence, `dot` recording | 1 (the style fix targets 1.18's `TrackedStyle`) |
| `SparseArrays` | weak (`SparseArraysExt`) | promote to the sparse style | stdlib |

Test-only: Zygote 0.7.18 (deferred cotangents, #19), Enzyme 0.13, Mooncake 0.5, ReverseDiff 1.18, LazyArrays 2,
DifferentiationInterface. Registered in the private registry `rcalxrc08/LocalRegistry`; FinancialFFT and
VibratoMonteCarlo depend on it.

Upstream workarounds (cite issue and version; recheck on every upstream release, remove when fixed):

| Workaround | Upstream issue | Affected versions | Last checked |
|:--|:--|:--|:--|
| `@noinline throw_length_error` | EnzymeAD/Enzyme.jl#4053 | Enzyme 0.13.219-0.13.220; fixed in 0.13.221, kept as good practice | 2026-11-19 |
| `@test_broken` Zygote cases with dense `ifelse` references | FluxML/Zygote.jl#1602 | Zygote 0.7 | 2026-11-19 |

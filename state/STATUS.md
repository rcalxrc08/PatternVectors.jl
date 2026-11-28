# Status

_What is in flight. Layer 4: volatile, rewritten freely; GitHub issues are the source of truth for tasks, this file is
the index an agent reads first._ Updated: 2026-11-28

## Current focus

Layered context (#22) on branch `context-layers`: done when `test/context.jl` and the full suite are green and the
branch is merged.

## Active work

| Item | Branch | Issue | Next step |
|:--|:--|:--|:--|
| Layered context | `context-layers` | #22 | review, merge, push |
| Reactant extension: PatternVectors reach XLA as fills, concatenations and broadcasts of the pattern values, never as `n` values (D-008) | `main` (bfb9ab9) | #24 | merged; strided sums for `dot` measured slower than the lazy reduction on the CPU, dropped |
| `dot` as a lazy reduction, Mooncake rule on `pattern_dot` (D-009) | `lazy-dot` | #26 | review; then FinancialFFT and VibratoMonteCarlo measure `dot` against `sum(w .* y)` again |

## Known issues

| Issue | Symptom | Workaround |
|:--|:--|:--|
| FluxML/Zygote.jl#1602 | Zygote mis-differentiates dense references with `ifelse` | `@test_broken` cases |
| local Enzyme on Julia 1.13 | `test_Enzyme.jl` aborted at line 37 locally (`handleKnownCallDerivatives`), not on CI | CI on 1.11 and 1.12 |
| Enzyme on CUDA | fails on the MX150 even for dense `CuArray`s; `MixedDuplicated` with a `PatternVector` | not verifiable on this machine |
| `dot` with a `SparseVector` | `MethodError`: ambiguous with SparseArrays' `dot(::AbstractVector, ::AbstractCompressedVector)` (on `main` before D-009 too) | `dot(collect(p), y)`; #25 |

## Technical debt

- `Project.toml` allows `Mooncake = "0.5"`, but `Mooncake.Nfwd` (used by the extension) exists only since 0.5.25: raise
  the floor.
- `pattern_to_vector_pullback` allocates a dense vector per pattern field for a dense cotangent; a lazy sum with the
  wrapped broadcast style removes them (tried 2026-10-28, not committed).
- Mooncake rule: a cap on K (differentiable arguments) in `_is_primitive`, above which the derived rule is faster
  (crossover not measured); deduplicating repeated arguments not implemented (`mooncake.md`).
- Possible EnzymeRule for `materialize_mixture` reusing the NDual approach (`autodiff.md`).
- Untracked local files: `src/04_patterns/symmetricpaddedfillpattern.jl`, `test/test_z*.jl`, `test/runbench.jl`,
  `test/bench_zygote_mooncake.jl`, `etc/logo2.svg`.

## Working notes

Notes of an active initiative live next to this file as `state/YYYY-MM-DD-<topic>.md`. When the initiative ends, move
what stays true into `references/` or `references/decisions.md` and delete the note.

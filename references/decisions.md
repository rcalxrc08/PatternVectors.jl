# Decisions

_Architecture decisions, oldest first. Append; a reversed decision gets a new entry that says "supersedes D-NNN"._
(Layer 3 reference.) Read before a refactor: a structure that looks accidental may be an entry here. Details stay in
the reference each entry points to.

## D-001 (2026-10-26, f3d96b7; simplified 2026-11-02, d217bc8) One Mooncake rule, on `materialize_mixture`

- **Decision:** Mooncake gets no rule for `PatternVector` (structural tangent) and one reverse rule for mixed
  broadcasts, hooked on `materialize_mixture`, recomputing `f` once per differentiable argument with an NDual seed.
- **Why:** three versions measured (`mooncake.md`): the recompute version is the fastest for cheap `f` on the CPU and
  in every measured case on the GPU, with 111 → 84 lines.
- **Consequences:** `materialize_mixture` stays the single hook; ChainRulesCore rules are never `@from_rrule`d.
- **Status:** active

## D-002 (2026-11-01, 2f923fc; 0.1.5) Fuse nested mixed broadcasts, fold scalar arithmetic

- **Decision:** mixed sub-expressions stay lazy and fuse; pattern-only ones are evaluated on the patterns; only plain
  scalar arithmetic is folded once.
- **Why:** eager mixed sub-expressions took 4x the memory (`a * real(s * y)`), nested scalars 67x (`a / (2 * pi)`).
- **Consequences:** the promises of `broadcast.md`, pinned by "Nested mixed broadcasts"; rule in `CLAUDE.md`.
- **Status:** active

## D-003 (2026-11-03, d20937f; 0.1.6) `dot` with broadcasts and a reduction only

- **Decision:** `dot` of PatternVectors is written with broadcasts and a reduction, never element indexing.
- **Why:** the generic `dot` indexes element by element and broke Zygote on `CuArray`s (#4).
- **Consequences:** `src/05_linearalgebra/`; ReverseDiff pairs in its extension (45a9299).
- **Status:** active

## D-004 (2026-11-14, 0d767ea; 0.1.8) Pullbacks unthunk; the `sum` rule covers every vector

- **Decision:** every pullback unthunks its cotangent; `rrule(sum, ::AbstractVector)` stays on every vector.
- **Why:** Zygote ≥ 0.7.17 thunks broke GPU gradients (#19); the O(1) `FillPattern` cotangent is up to 8x faster on the
  CPU and 2x on the GPU than Zygote's own rule (#20).
- **Consequences:** rule in `CLAUDE.md`; `autodiff.md`.
- **Status:** active

## D-005 (2026-11-14, 709f3fc; 0.1.9) Error paths out of line

- **Decision:** throws inside functions that run in differentiated code are `@noinline` functions.
- **Why:** the inlined length check allocated in every broadcast `materialize`; Enzyme ≥ 0.13.219 aborted on it
  (EnzymeAD/Enzyme.jl#4053, #15). Fixed upstream in 0.13.221; kept as good practice.
- **Consequences:** rule in `CLAUDE.md`.
- **Status:** active

## D-006 (2026-11-14, 28fe6d7; 0.1.9) Own `rrule` for `dot`

- **Decision:** `dot_rrule` for `dot` with a `PatternVector`: no `w .* y` forward, a `PatternVector` cotangent.
- **Why:** 1.3-1.75x less memory and 1.2-1.5x faster on the GPU at n = 10^6 (#21); ChainRules' generic rule gained
  nothing over `sum(w .* y)`.
- **Consequences:** FinancialFFT and VibratoMonteCarlo measured neutral and keep `sum` (`autodiff.md`).
- **Status:** active

## D-007 (2026-11-22) Layered context (ICM)

- **Decision:** the repository context follows the Interpretable Context Methodology: `CLAUDE.md` (L0), `CONTEXT.md`
  routing (L1), stage contracts (L2), `references/` (L3), `state/` (L4); `test/context.jl` keeps it in sync (#22).
- **Why:** the same layout worked in FinancialCalibrator and VibratoMonteCarlo; a single `CLAUDE.md` loads everything
  for every task and drifts.
- **Consequences:** every source file belongs to a stage folder and its contract; `MOONCAKE_NOTES.md` became
  `references/mooncake.md`.
- **Status:** active

## D-008 (2026-11-27) PatternVectors reach XLA as structure, never as n values

- **Decision:** `ReactantExt` hands Reactant a PatternVector built from fills, concatenations and broadcasts of its
  pattern values (`traced_array`), numbers behind an `optimization_barrier`; `getindex_pattern` takes any index type
  (a traced one included).
- **Why:** Reactant's default turns any unknown array into a dense constant of its `n` values, which defeats the
  package; Reactant (below 1024 elements) and XLA (concatenations, comparisons with an iota) fold constant
  expressions back into one unless the scalars are opaque (`reactant.md`). Compile time stays constant in `n`
  (0.6 s against 15-35 s for a dense constant at 4·10^6), run time equal on CUDA.
- **Consequences:** a new pattern needs a `traced_array` method; `test/reactant/runtests.jl` fails when a constant of
  `n` elements appears. On XLA CPU the structured weights of a reduction run 2-5x slower than a dense constant
  (YNNPACK fusion), accepted for the compile time and the program size.
- **Status:** active

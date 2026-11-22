# CLAUDE.md

**PatternVectors.jl** provides `PatternVector`, an immutable, `isbits` `AbstractVector` that stores its length and a few
pattern values instead of every element. Broadcasts are the point of the package: an expression on PatternVectors only
is evaluated once per distinct value and returns a `PatternVector`; mixed with a dense array it fuses into one loop on the
array's device; Zygote, Mooncake, Enzyme and ReverseDiff accumulate gradients on the pattern values. FinancialFFT and
VibratoMonteCarlo use it for quadrature weights. Registered in the private registry `rcalxrc08/LocalRegistry`, not in
General. Goals and non-goals: `references/mission.md`; why things are the way they are: `references/decisions.md`.

This workspace follows the Interpretable Context Methodology (layered context, arXiv:2603.16021): load only what the
task needs.

| Layer | Where | What |
|:--|:--|:--|
| 0 | this file | identity, map, rules that hold everywhere |
| 1 | `CONTEXT.md` | which stage or folder a task belongs to, shared references |
| 2 | `src/0N_*/CONTEXT.md`, `ext/`, `test/`, `docs/`, `benchmark/`, `etc/` | a contract per stage: inputs, process, outputs, runtime checks |
| 3 | `references/*.md` | stable facts, decisions and measurements, loaded when a contract lists them |
| 4 | `state/STATUS.md` | what is in flight: current focus, open issues, known problems |

Startup: read `state/STATUS.md`, then route the task through `CONTEXT.md`. Runtime truth comes from a Kaimon session
for this project (`kaimon.toml` pins the Julia); each contract's Verify table lists the checks to run after a change.

Map: `src/PatternVectors.jl` includes the stages in order, `01_vector` → `02_broadcast` → `03_rrules` → `04_patterns` →
`05_linearalgebra`; `ext/` the Mooncake, ReverseDiff and SparseArrays extensions; `test/` the suite; `docs/` the
Documenter site; `benchmark/` the CI benchmark suite; `etc/` the README benchmark script and the logo.

Rules for every change (details in `references/conventions.md`):

- Any number type, any array type (CPU and GPU): no `Float64` buffers, no index loops; a `PatternVector` stays `isbits`.
- The broadcast promises of `references/broadcast.md` hold: pattern-only sub-expressions on the patterns, mixed ones
  lazy and fused, scalar arithmetic folded once.
- Pullbacks unthunk their cotangent; error paths inside differentiated code are out of line (`@noinline`).
- Reuse the author's packages and established libraries before writing new code; `@muladd` for every `a * b + c`.
- Format with `.JuliaFormatter.toml`; update the CONTEXT.md, references and `state/STATUS.md` a change touches in the
  same commit (a new rule also gets an entry in `references/decisions.md`); default branch `main`; push to `origin`
  (GitHub) only.

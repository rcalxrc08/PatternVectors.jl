# CONTEXT.md: task routing (Layer 1)

Find the task, open the contract it points to, and load only the inputs listed there.

| Task | Contract |
|:--|:--|
| The `PatternVector` type, the pattern interface, `getindex`, ranges, `sum` | `src/01_vector/CONTEXT.md` |
| Broadcast styles, pattern-only and mixed broadcasts, fusion, `materialize_mixture` | `src/02_broadcast/CONTEXT.md` |
| ChainRulesCore rules (Zygote): the constructor, `sum` | `src/03_rrules/CONTEXT.md` |
| A pattern (`FillPattern`, `EvenOddPattern`, …) or a new one | `src/04_patterns/CONTEXT.md` |
| `dot` and its rules | `src/05_linearalgebra/CONTEXT.md` |
| Mooncake rule, ReverseDiff and SparseArrays styles | `ext/CONTEXT.md` |
| Adding or running tests, AD backends in tests, CUDA tests | `test/CONTEXT.md` |
| Documentation pages, figures, the shared site design | `docs/CONTEXT.md` |
| The CI benchmark suite | `benchmark/CONTEXT.md` |
| The README benchmark table, the logo | `etc/CONTEXT.md` |
| Commands, CI, releases, sub-environments | `references/workflows.md` |
| What is being worked on, known issues | `state/STATUS.md` |
| Whether a change fits the package's goals; why a rule exists | `references/mission.md`, `references/decisions.md` |

Shared references (Layer 3), each loaded only when a contract lists it: `public_api.md`, `conventions.md`,
`broadcast.md`, `autodiff.md`, `mooncake.md`, `dependencies.md`, `workflows.md`, `testing.md`, `performance.md`,
`docs_design.md`.

The stages communicate through Julia values: 01 defines `PatternVector{T, P}` and the pattern interface; 02 makes
broadcasts return a `PatternVector` (pattern-only) or the dense array type (mixed); 03 gives Zygote O(1) cotangents;
04 implements the interface per pattern; 05 reduces with `dot` on the device of the dense argument.

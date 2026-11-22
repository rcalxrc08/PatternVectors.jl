# Conventions

_Coding standards that hold in every stage._ (Layer 3 reference.)

- Any number type and any array type: no `Float64` buffers or literals, no index loops (GPU arrays forbid scalar
  indexing); results built from the types of the inputs.
- A `PatternVector` stays `isbits` for `isbits` pattern values; all fields of a pattern have the element type.
- Pullbacks unthunk their cotangent; error paths of functions that run inside differentiated code are `@noinline`
  (`decisions.md` D-004, D-005).
- Branches on a differentiated path are allowed (Zygote ≥ 0.7.19, which the tests require, fixed their penalty,
  FluxML/Zygote.jl#1754); as a select, `max`/`min` or sign weights rather than `ifelse` on a scalar flag.
- `@muladd` (MuladdMacro.jl) for every `a * b + c` (not a dependency yet: add it before the first use).
- Reuse before writing: the author's packages, then established libraries.
- Formatting: JuliaFormatter with `.JuliaFormatter.toml` (`margin = 1200`, `whitespace_typedefs = true`,
  `trailing_comma = false`): `julia -e 'using JuliaFormatter; format(".")'`.
- Markdown tables in the docs need `|:--|`: plain `|---|` columns are right-aligned.

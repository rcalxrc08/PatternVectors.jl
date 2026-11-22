# Broadcast

_The two styles and the promises of `materialize_if_needed`._ (Layer 3 reference, stage 02.)

## Styles

- `ArrayStylePatternVector{P}`: PatternVectors and scalars only. `materialize_pattern` evaluates the function on the
  pattern values and returns a `PatternVector`.
- `PatternMixtureArrayStyle{N, T}`: a PatternVector mixed with a dense array of style `T` (`Array`, `CuArray`, …). The
  result has the type of the dense array.

A mixed broadcast goes through `materialize_if_needed` on its arguments, recursively, then through
`Broadcast.flatten` and `materialize_mixture(style, f, args...)`, the single hook for AD rules.

## The promises (keep them)

- A sub-expression on PatternVectors only is evaluated on the patterns, never on a full array: `f.(s) .* x` calls `f`
  once per distinct value.
- A mixed or dense sub-expression stays lazy, so everything fuses in one loop. Computing it on its own allocates one
  array per level of nesting (`a * real(s * y)` took 4 times the memory up to 0.1.4).
- Plain arithmetic on scalars (`ScalarArithmetic`) is evaluated once: `Broadcast.flatten` allocates for every element on
  a nested scalar like `a / (2 * pi)` (67 times the memory up to 0.1.4). Do not extend this to arbitrary functions,
  they can be impure: as in Base, `rand()` in a broadcast must stay element by element.
- In-place broadcasts (`d .= …`) do not go through this path.

The tests in "Nested mixed broadcasts" (`test/runtests.jl`) pin these rules, memory included. They fail when mixed
sub-expressions are computed eagerly, so a change there must keep them.

`get_style` unwraps nested `PatternMixtureArrayStyle`s (they appear when mixing with `CuArrayStyle`), so
`materialize_mixture` always gets the innermost style.

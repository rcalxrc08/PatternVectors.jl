# CLAUDE.md

Guidance for working in this repository.

## What this is

`PatternVectors.jl` provides `PatternVector`, an immutable `AbstractVector` that stores its length and a few pattern values instead of every element. Broadcasts are the point of the package: an expression on PatternVectors only is evaluated once per distinct value and returns a `PatternVector`, and it works with Zygote, Mooncake, Enzyme and ReverseDiff, with gradients accumulated on the pattern values.

The registered versions are in the private registry `rcalxrc08/LocalRegistry`, not in General. FinancialFFT and VibratoMonteCarlo depend on them.

## Commands

```bash
julia --project -e 'import Pkg; Pkg.test()'             # test/runtests.jl, then the AD backends
julia --project=benchmark benchmark/runbenchmarks.jl
julia --project=docs docs/make.jl                      # Documenter site, into docs/build
julia -e 'using JuliaFormatter; format(".")'            # uses .JuliaFormatter.toml
```

CI runs on Julia 1.11 and 1.12. The default branch is `main`.

## Architecture

- [src/patternvector.jl](src/patternvector.jl): the type, `getindex`, ranges, `sum`.
- [src/patterns/](src/patterns/): one file per pattern. A new pattern only needs the methods listed in "Defining a new pattern" in the README, and can live outside the package.
- [src/broadcast.jl](src/broadcast.jl): see below.
- [src/rrules.jl](src/rrules.jl): ChainRulesCore rules (Zygote).
- [src/linearalgebra.jl](src/linearalgebra.jl): `dot` with broadcasts and a reduction only, so that it runs on the device of a dense argument (the generic `dot` indexes element by element). ChainRules reaches it in reverse mode: the `rrule` of `sum` returns a `PatternVector` cotangent, and the cotangent of a scalar times an array is `dot(Ȳ, B)` (issue #4).
- [docs/](docs/): Documenter site, published with GitHub Pages from Actions (DeployPage.yml). Markdown tables need `|:--|`: plain `|---|` columns are right-aligned. The logo and favicon live in etc/ and are copied by make.jl.
- [ext/](ext/): `MooncakeExt` (see [MOONCAKE_NOTES.md](MOONCAKE_NOTES.md)), `ReverseDiffExt`, `SparseArraysExt`.

### Broadcasting

There are two styles:
- `ArrayStylePatternVector{P}`: PatternVectors and scalars only. `materialize_pattern` evaluates the function on the pattern values and returns a `PatternVector`.
- `PatternMixtureArrayStyle{N,T}`: a PatternVector mixed with a dense array of style `T` (`Array`, `CuArray`, ...). The result has the type of the dense array.

A mixed broadcast goes through `materialize_if_needed` on its arguments, recursively, then through `Broadcast.flatten` and `materialize_mixture(style, f, args...)`, the single hook for AD rules. The rules of `materialize_if_needed` are what the package promises, keep them:
- A sub-expression on PatternVectors only is evaluated on the patterns, never on a full array: `f.(s) .* x` calls `f` once per distinct value.
- A mixed or dense sub-expression stays lazy, so everything fuses in one loop. Computing it on its own allocates one array per level of nesting (`a * real(s * y)` took 4 times the memory up to 0.1.4).
- Plain arithmetic on scalars (`ScalarArithmetic`) is evaluated once: `Broadcast.flatten` allocates for every element on a nested scalar like `a / (2 * pi)` (67 times the memory up to 0.1.4). Do not extend this to arbitrary functions, they can be impure: as in Base, `rand()` in a broadcast must stay element by element.
- In-place broadcasts (`d .= ...`) do not go through this path.

The tests in "Nested mixed broadcasts" pin these rules, memory included. They fail when mixed sub-expressions are computed eagerly, so a change there must keep them.

## Tests

`test/runtests.jl` holds the patterns and the broadcasts, then runs `test_Zygote.jl`, `test_ReverseDiff.jl`, `test_Mooncake.jl` and `test_Enzyme.jl`. `cuda_tests.jl` needs a GPU and is not run on CI. Some Zygote cases are pinned with `@test_broken`.

Compare allocations with `@allocated` on a function called once before, inside a function: at the top level the globals inflate them.

## Conventions

- Formatting: JuliaFormatter with `margin = 1200`, `whitespace_typedefs = true`, `trailing_comma = false`.
- Everything must work for any number type and on GPU arrays: no `Float64` buffers, no index loops, and a `PatternVector` stays `isbits`.

## Releasing

Bump `version` in `Project.toml`, push to `main`, wait for CI, then register the commit on LocalRegistry with `LocalRegistry.register`. Commits on the registry are named `New version: PatternVectors vX.Y.Z`.

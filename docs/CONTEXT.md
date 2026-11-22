# Documentation

The Documenter site, published with GitHub Pages from Actions (`DeployPage.yml`).

## Inputs

| Load | Why |
|:--|:--|
| `make.jl` | pages, `checkdocs = :exports`, assets copied from `etc/` |
| `figures.jl` | the home page figures (every pattern as a strip of cells), the gradient chart of `performance.md`, `hero_stats()` |
| `surface.jl` | palette and colour helpers used by `figures.jl` |
| `references/docs_design.md` | the shared site design, table alignment, what is static data |
| `references/public_api.md` | every exported name is documented |
| `references/workflows.md` | how to build the docs locally |

## Process

- The design files are identical copies across the rcalxrc08 packages: edit them together (`references/docs_design.md`).
- The gradient chart of `performance.md` is static data in `figures.jl`: update the table and the chart together.
- A changed signature updates its docstring, the page that shows it and `references/public_api.md` in the same commit.

## Outputs

A site that builds with no warnings; doctests pass.

## Verify (Kaimon)

| Check | Expect |
|:--|:--|
| `ex`: `using Documenter, PatternVectors; doctest(PatternVectors; manual = false)` | pass |

# Documentation design

_The site design shared by the docs of all the rcalxrc08 packages._ (Layer 3 reference.)

- Documenter site, published with GitHub Pages from Actions (`DeployPage.yml`).
- Identical copies across packages, edit them together: `docs/src/assets/style.css` (Dark Quant palette), `dark.js`,
  `figures.css`, `site.css` (page headers with kicker, lede and section links, numbered sections, cards) and `home.css`
  (landing page).
- `index.md` is the landing page; `docs/figures.jl` (a module of its own in `make.jl`) draws its hero (every pattern as
  a strip of cells, read from the vectors; palette and colour helpers from `docs/surface.jl`), the gradient chart of
  `performance.md`'s table (static data: update both together) and `hero_stats()` for the stat tiles; figures go to
  `docs/src/assets/figures/` (git-ignored).
- The logo and favicon live in `etc/` and are copied by `make.jl`.
- Markdown tables need `|:--|`: plain `|---|` columns are right-aligned.

using Documenter, PatternVectors
# Documenter takes the logo and the favicon from the assets: etc/ holds the only copies, and git keeps no empty folder.
assets = mkpath(joinpath(@__DIR__, "src", "assets"))
for asset in ("logo.png", "favicon.ico")
    cp(joinpath(@__DIR__, "..", "etc", asset), joinpath(assets, asset), force = true)
end
# The figures and the key numbers of the home page are computed with the package, in a light and a dark variant (see
# figures.jl and assets/figures.css), in a module of their own.
module DocumentationFigures
include(joinpath(@__DIR__, "figures.jl"))
end
DocMeta.setdocmeta!(PatternVectors, :DocTestSetup, :(using PatternVectors); recursive = true)
makedocs(format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true", assets = ["assets/favicon.ico", "assets/figures.css", "assets/style.css", "assets/site.css", "assets/home.css", "assets/dark.js"], edit_link = "main"), repo = Remotes.GitHub("rcalxrc08", "PatternVectors.jl"), sitename = "PatternVectors.jl", modules = [PatternVectors], checkdocs = :exports, pages = ["Home" => "index.md", "Guides" => ["Getting started" => "usage.md", "Automatic differentiation" => "ad.md", "Performance" => "performance.md", "New patterns" => "patterns.md"], "Reference" => ["API" => "api.md"]])

using Documenter, PatternVectors
# Documenter takes the logo and the favicon from the assets: etc/ holds the only copies, and git keeps no empty folder.
assets = mkpath(joinpath(@__DIR__, "src", "assets"))
for asset in ("logo.png", "favicon.ico")
    cp(joinpath(@__DIR__, "..", "etc", asset), joinpath(assets, asset), force = true)
end
DocMeta.setdocmeta!(PatternVectors, :DocTestSetup, :(using PatternVectors); recursive = true)
makedocs(
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true", assets = ["assets/favicon.ico", "assets/style.css"], edit_link = "main"),
    repo = Remotes.GitHub("rcalxrc08", "PatternVectors.jl"),
    sitename = "PatternVectors.jl",
    modules = [PatternVectors],
    checkdocs = :exports,
    pages = ["index.md", "usage.md", "ad.md", "performance.md", "patterns.md", "api.md"],
)

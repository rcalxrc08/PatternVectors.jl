# Figures of the home page, computed with the package at build time and saved as SVG in src/assets/figures (git-ignored),
# in a light and a dark variant: assets/figures.css shows the one of the theme of the reader. The palette is the one of
# the other rcalxrc08 packages (surface.jl holds it, with the colour helpers).
using PatternVectors, Plots, Printf
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
gr()
include(joinpath(@__DIR__, "surface.jl"))

const FIGURES = mkpath(joinpath(@__DIR__, "src", "assets", "figures"))
const THEMES = (light = (series = ["#2563d9", "#0a8a6f", "#b86e0b", "#0b7fb3", "#2f8a3c", "#c2456e"], text = "#1f2328", muted = "#5b6470", axis = "#b9b4ab", grid = "#e6e3de", diverging = ["#2563d9", "#f7f6f3", "#c47a12"]), dark = (series = ["#4f8cff", "#00d4aa", "#ffb84d", "#38cfff", "#7ee787", "#ff6b9d"], text = "#f5f7fa", muted = "#a0aec0", axis = "#3a4466", grid = "#252d47", diverging = ["#4f8cff", "#151b2e", "#ffb84d"]))

# The hero: every pattern of the package as a strip of cells, each cell coloured by the value of the element, read from
# the PatternVector itself; on the right, what the vector stores.
const HERO_PATTERNS = ["ZeroPattern" => PatternVector(20, ZeroPattern(0.0)), "FillPattern" => PatternVector(20, FillPattern(0.7)), "InitialValuePattern" => PatternVector(20, InitialValuePattern(1.0, 0.35)), "FinalValuePattern" => PatternVector(20, FinalValuePattern(0.35, 1.0)), "PaddedFillPattern" => PatternVector(20, PaddedFillPattern(0.15, 0.6, 1.0)), "EvenOddPattern" => PatternVector(20, EvenOddPattern(0.2, 0.9)), "PaddedEvenOddPattern" => PatternVector(20, PaddedEvenOddPattern(1 / 3, 4 / 3, 2 / 3, 1 / 3))]

function hero_svg(theme::Symbol)
    v = SURFACE_THEMES[theme]
    W, H = 960, 640
    x0, y0, cell, gap, row = 250, 70, 26, 4, 74
    io = IOBuffer()
    print(io, "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 $W $H\" font-family=\"system-ui, -apple-system, Segoe UI, Roboto, sans-serif\" role=\"img\" aria-label=\"The patterns of PatternVectors.jl, each a vector of 20 elements coloured by value\">")
    print(io, "<text x=\"$(x0)\" y=\"40\" fill=\"$(v.muted)\" font-size=\"15\">20 elements each, read from the PatternVector</text>")
    print(io, "<text x=\"$(W - 20)\" y=\"40\" fill=\"$(v.muted)\" font-size=\"15\" text-anchor=\"end\">distinct values</text>")
    for (k, (name, x)) in enumerate(HERO_PATTERNS)
        y = y0 + (k - 1) * row
        lo, hi = 0.0, 4 / 3
        print(io, "<text x=\"$(x0 - 18)\" y=\"$(y + 18)\" fill=\"$(v.text)\" font-size=\"16\" font-family=\"ui-monospace, Cascadia Code, Consolas, monospace\" text-anchor=\"end\">$name</text>")
        for i in eachindex(x)
            c = x[i] == 0 ? v.floor : surface_ramp(v.ramp, (x[i] - lo) / (hi - lo))
            print(io, "<rect x=\"$(x0 + (i - 1) * (cell + gap))\" y=\"$y\" width=\"$cell\" height=\"$cell\" rx=\"5\" fill=\"$c\" stroke=\"$(v.axis)\" stroke-width=\"$(x[i] == 0 ? 1 : 0)\"/>")
        end
        distinct = length(unique(collect(x)))
        print(io, "<text x=\"$(W - 20)\" y=\"$(y + 18)\" fill=\"$(v.marker)\" font-size=\"17\" font-weight=\"700\" text-anchor=\"end\" font-family=\"ui-monospace, Cascadia Code, Consolas, monospace\">$distinct</text>")
    end
    print(io, "</svg>")
    return String(take!(io))
end
for theme in (:dark, :light)
    write(joinpath(FIGURES, "hero-patterns-$theme.svg"), hero_svg(theme))
end

function base_plot(theme; kwargs...)
    return plot(; size = (760, 430), background_color = :transparent, foreground_color_text = theme.muted, foreground_color_guide = theme.muted, foreground_color_axis = theme.axis, foreground_color_border = theme.axis, foreground_color_legend = nothing, legend_background_color = :transparent, legendfontcolor = theme.text, titlefontcolor = theme.text, gridcolor = theme.grid, gridalpha = 1, gridlinewidth = 1, framestyle = :axes, fontfamily = "sans-serif", titlefontsize = 12, guidefontsize = 10, tickfontsize = 9, legendfontsize = 9, titlelocation = :left, left_margin = 4Plots.mm, bottom_margin = 4Plots.mm, kwargs...)
end

# The gradient timings of performance.md (etc/benchmarks.jl, n = 10^5, Intel i7-8550U, one thread), in microseconds.
const BACKENDS = ["Forward", "Zygote", "Mooncake", "Enzyme"]
const GRADIENTS = ["sum(w .* x)" => ([387, 1860, 11600, 1680], [165, 959, 864, 794]), "sum(sin.(w .* x) .* exp.(w))" => ([2770, 7400, 18800, 9810], [1640, 4690, 12700, 9000]), "sum(exp.(w) .* w)" => ([1270, 4430, 12800, 4800], [0.067, 2.2, 0.671, 0.228])]

function gradients_figure(theme)
    panels = map(enumerate(GRADIENTS)) do (k, (name, (dense, pattern)))
        p = base_plot(theme; title = name, titlefontsize = 10, yscale = :log10, ylabel = k == 1 ? "time of a gradient (μs)" : "", legend = k == 1 ? :topleft : false, xticks = (1:4, BACKENDS), xrotation = 25, ylims = (0.03, 1e5), yticks = (10.0 .^ (-1:2:5), ["0.1", "10", "1 000", "100 000"]), left_margin = 3Plots.mm)
        bar!(p, (1:4) .- 0.2, dense; bar_width = 0.38, color = theme.series[1], linewidth = 0, label = "dense Vector", fillrange = 0.03)
        bar!(p, (1:4) .+ 0.2, pattern; bar_width = 0.38, color = theme.series[2], linewidth = 0, label = "PatternVector", fillrange = 0.03)
        p
    end
    return plot(panels...; layout = (1, 3), size = (960, 400), background_color = :transparent)
end
for (name, theme) in pairs(THEMES)
    savefig(gradients_figure(theme), joinpath(FIGURES, "gradients-$name.svg"))
end

"The key numbers of the home page, computed when the documentation is built."
function hero_stats()
    n = 10^6
    w = PatternVector(n, PaddedEvenOddPattern(1.0, 4.0, 2.0, 1.0))
    tile(v, u, l; tone = "") = "<div class=\"fc-stat $tone\"><div class=\"fc-stat-value\">$v<span>$u</span></div><div class=\"fc-stat-label\">$l</div></div>"
    tiles = [tile("$(sizeof(w))", " bytes", "a PatternVector of 10⁶ Simpson-like weights, against $(sizeof(collect(w)) ÷ 10^6) MB dense"), tile(isbits(w) ? "isbits" : "boxed", "", "so it goes to a GPU kernel as is"; tone = "is-teal"), tile("19 000", "×", "faster ForwardDiff gradient of sum(exp.(w) .* w) at n = 10⁵: 67 ns against 1.27 ms"; tone = "is-amber"), tile("O(1)", "", "sum and range indexing, whatever the length")]
    return HTML("<div class=\"fc-stats\">" * join(tiles) * "</div>")
end

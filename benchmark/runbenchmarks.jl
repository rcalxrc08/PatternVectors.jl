# Runs the suite of benchmarks.jl and prints a summary of PatternVector against dense vectors.
#   julia --project=benchmark benchmark/runbenchmarks.jl
# Tuning parameters are read from (or, when missing or outdated, written to) benchmark/tune.json; results go to
# benchmark/results.json and benchmark/results.md (and to the GitHub job summary when run in GitHub Actions).
using BenchmarkTools, Printf, Statistics

BenchmarkTools.DEFAULT_PARAMETERS.seconds = parse(Float64, get(ENV, "BENCHMARK_SECONDS", "0.5"))
include(joinpath(@__DIR__, "benchmarks.jl"))

const TUNE_FILE = joinpath(@__DIR__, "tune.json")
const RESULTS_FILE = joinpath(@__DIR__, "results.json")
const REPORT_FILE = joinpath(@__DIR__, "results.md")

function load_or_tune!(suite)
    if isfile(TUNE_FILE)
        try
            loadparams!(suite, BenchmarkTools.load(TUNE_FILE)[1], :evals, :samples)
            println("Loaded tuning parameters from $TUNE_FILE")
            return suite
        catch e
            println("Tuning parameters in $TUNE_FILE do not match the suite ($(typeof(e))): tuning again")
        end
    end
    tune!(suite)
    BenchmarkTools.save(TUNE_FILE, params(suite))
    println("Saved tuning parameters to $TUNE_FILE")
    return suite
end

load_or_tune!(SUITE)
results = run(SUITE; verbose = false)
BenchmarkTools.save(RESULTS_FILE, results)
estimates = median(results)

# (path without the last key) => Dict("PatternVector" => estimate, "dense" => estimate)
pairs_by_case = Dict{Vector{Any}, Dict{String, BenchmarkTools.TrialEstimate}}()
for (path, est) in BenchmarkTools.leaves(estimates)
    get!(pairs_by_case, path[1:(end - 1)], Dict{String, BenchmarkTools.TrialEstimate}())[path[end]] = est
end
speedup(p) = time(p["dense"]) / time(p["PatternVector"])
geomean(v) = isempty(v) ? NaN : exp(mean(log.(v)))
fmt_speedup(s) = isnan(s) ? "–" : s >= 100 ? @sprintf("%.0f×", s) : @sprintf("%.1f×", s)
sorted_cases(f) = sort!([(k, v) for (k, v) in pairs_by_case if f(k)]; by = first)

io = IOBuffer()
println(io, "## PatternVectors.jl benchmarks\n")
println(io, "n = $N, median times; speedup = dense time / PatternVector time (geometric mean over the functions ",
    join(first.(FUNCTIONS), ", "), " and the operations ", join(["`$(first(o))`" for o in OPERATIONS], ", "), ", `sum(p)`).\n")

tnames = first.(ELTYPES)
pnames = first.(PATTERNS)
println(io, "### Forward\n")
println(io, "| Pattern | ", join(tnames, " | "), " |")
println(io, "|---|", repeat("---|", length(tnames)))
for pname in pnames
    cells = map(tnames) do tname
        fmt_speedup(geomean([speedup(p) for (k, p) in pairs_by_case if k[1] == "forward" && k[3] == pname && k[4] == tname]))
    end
    println(io, "| $pname | ", join(cells, " | "), " |")
end

gcols = [(backend, op) for backend in ("Zygote", "Mooncake") for op in first.(GRADIENT_OPERATIONS)]
println(io, "\n### Gradients (Float64, w.r.t. p and x)\n")
println(io, "| Pattern | ", join(["$b `$o`" for (b, o) in gcols], " | "), " |")
println(io, "|---|", repeat("---|", length(gcols)))
for pname in pnames
    cells = [haskey(pairs_by_case, Any["gradient", b, o, pname]) ? fmt_speedup(speedup(pairs_by_case[Any["gradient", b, o, pname]])) : "–" for (b, o) in gcols]
    println(io, "| $pname | ", join(cells, " | "), " |")
end

println(io, "\n### All benchmarks\n")
println(io, "| Benchmark | PatternVector | dense | speedup | PatternVector allocations |")
println(io, "|---|---|---|---|---|")
for (k, p) in sorted_cases(_ -> true)
    pv, de = p["PatternVector"], p["dense"]
    println(io, "| ", join(k, " / "), " | ", BenchmarkTools.prettytime(time(pv)), " | ", BenchmarkTools.prettytime(time(de)),
        " | ", fmt_speedup(speedup(p)), " | ", allocs(pv), " (", BenchmarkTools.prettymemory(memory(pv)), ") |")
end
if !isempty(SKIPPED)
    println(io, "\n### Skipped combinations\n")
    foreach(s -> println(io, "- ", s), SKIPPED)
end

report = String(take!(io))
print(report)
write(REPORT_FILE, report)
haskey(ENV, "GITHUB_STEP_SUMMARY") && open(io -> write(io, report), ENV["GITHUB_STEP_SUMMARY"], "a")
println("\nSaved results to $RESULTS_FILE and $REPORT_FILE")

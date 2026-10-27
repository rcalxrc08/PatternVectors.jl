# Benchmark suite of PatternVectors.jl: every pattern, with several element types and functions, against the same
# operation on the dense vector (`collect`), plus gradients with Zygote and Mooncake.
# Defines `SUITE` (a BenchmarkTools.BenchmarkGroup) and `SKIPPED` (combinations that cannot be built).
using PatternVectors, BenchmarkTools, Zygote, Mooncake, ForwardDiff

const N = 10_000

# Four values of each element type, used as the pattern values
const ELTYPES = (
    "Float16" => Float16.((0.3, 1.2, 0.7, 0.4)),
    "Float64" => (0.3, 1.2, 0.7, 0.4),
    "Irrational" => (π, π, π, π),
    "Dual" => ForwardDiff.Dual.((0.3, 1.2, 0.7, 0.4), 1.0),
)

const PATTERNS = (
    "ZeroPattern" => v -> ZeroPattern(v[1]),
    "FillPattern" => v -> FillPattern(v[1]),
    "InitialValuePattern" => v -> InitialValuePattern(v[1], v[2]),
    "FinalValuePattern" => v -> FinalValuePattern(v[1], v[2]),
    "PaddedFillPattern" => v -> PaddedFillPattern(v[1], v[2], v[3]),
    "EvenOddPattern" => v -> EvenOddPattern(v[1], v[2]),
    "PaddedEvenOddPattern" => v -> PaddedEvenOddPattern(v[1], v[2], v[3], v[4]),
)

poly(u) = u * u + 2 * u + 1
const FUNCTIONS = ("sin" => sin, "exp" => exp, "poly" => poly)

# Forward operations on a vector p (PatternVector or dense) and a dense Float64 vector x
const OPERATIONS = (
    "f.(p)" => (f, p, x) -> f.(p),
    "sum(f.(p) .* x)" => (f, p, x) -> sum(f.(p) .* x),
)

# Gradients (Float64) w.r.t. p and x
const GRADIENT_OPERATIONS = (
    "sum(sin.(p) .* x)" => (p, x) -> sum(sin.(p) .* x),
    "sum(exp.(p) .* p)" => (p, x) -> sum(exp.(p) .* p),
)

const SUITE = BenchmarkGroup()
const SKIPPED = String[]

# Group at the path `keys` of `g`, created when missing
function group(g::BenchmarkGroup, keys...)
    for k in keys
        g = haskey(g, k) ? g[k] : (g[k] = BenchmarkGroup())
    end
    return g
end
firstline(e) = first(split(sprint(showerror, e), '\n'))

x = rand(N)
for (tname, vals) in ELTYPES, (pname, make) in PATTERNS
    pv, dense = try
        pv = PatternVector(N, make(vals))
        pv, collect(pv)
    catch e
        push!(SKIPPED, "$pname{$tname}: $(firstline(e))")
        continue
    end
    g = group(SUITE, "forward", "sum(p)", pname, tname)
    g["PatternVector"] = @benchmarkable sum($pv)
    g["dense"] = @benchmarkable sum($dense)
    for (oname, op) in OPERATIONS, (fname, f) in FUNCTIONS
        try
            op(f, pv, x) ≈ op(f, dense, x) || error("PatternVector and dense results differ")
        catch e
            push!(SKIPPED, "$oname with $fname on $pname{$tname}: $(firstline(e))")
            continue
        end
        g = group(SUITE, "forward", oname, pname, tname, fname)
        g["PatternVector"] = @benchmarkable $op($f, $pv, $x)
        g["dense"] = @benchmarkable $op($f, $dense, $x)
    end
end

for (pname, make) in PATTERNS, (oname, op) in GRADIENT_OPERATIONS
    pv = PatternVector(N, make(last(ELTYPES[2])))
    dense = collect(pv)
    for (vname, v) in (("PatternVector", pv), ("dense", dense))
        group(SUITE, "gradient", "Zygote", oname, pname)[vname] = @benchmarkable Zygote.gradient($op, $v, $x)
        rule = Mooncake.build_rrule(op, v, x)
        group(SUITE, "gradient", "Mooncake", oname, pname)[vname] = @benchmarkable Mooncake.value_and_gradient!!($rule, $op, $v, $x)
    end
end

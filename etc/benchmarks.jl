# Benchmarks of the README: dense weights vs PatternVector, forward and gradients (Zygote, Mooncake, Enzyme).
# Run with PatternVectors, BenchmarkTools, Zygote, Mooncake and Enzyme in the active environment.
using PatternVectors, BenchmarkTools, Zygote, Mooncake, Enzyme, Printf

# Weights [a, b, c, b, c, ..., d] (Simpson-like), dense vs PatternVector, built from the same parameters
dense_w(a, b, c, d, idx) = [a, b, c, d][idx]
pattern_w(a, b, c, d, idx) = PatternVector(length(idx), PaddedEvenOddPattern(a, b, c, d))
function make_idx(n)
    idx = [isodd(i) ? 3 : 2 for i in 1:n]
    idx[1] = 1
    idx[end] = 4
    return idx
end

# (name, uses x, function of the weights builder W). Gradients are taken w.r.t. a, b, c, d and x (when used);
# idx only carries the length and the layout of the dense weights.
cases = (
    ("`sum(w .* x)`", true, W -> (a, b, c, d, x, idx) -> sum(W(a, b, c, d, idx) .* x)),
    ("`sum(sin.(w .* x) .* exp.(w))`", true, W -> (a, b, c, d, x, idx) -> (w = W(a, b, c, d, idx); sum(sin.(w .* x) .* exp.(w)))),
    ("`sum(exp.(w) .* w)`", false, W -> (a, b, c, d, idx) -> (w = W(a, b, c, d, idx); sum(exp.(w) .* w))),
)

fmt(t) = t < 1e-6 ? @sprintf("%.0f ns", t * 1e9) : t < 1e-3 ? @sprintf("%.1f µs", t * 1e6) : @sprintf("%.2f ms", t * 1e3)
grads_close(g1, g2) = all(map((u, v) -> isapprox(u, v; rtol = 1e-8), g1, g2))

# Gradients w.r.t. the first K arguments (the last argument, idx, is constant)
zyg(f, args, K) = map((g, a) -> g === nothing ? zero(a) : g, Zygote.gradient(f, args...)[1:K], args[1:K])
moon(rule, f, args, K) = Mooncake.value_and_gradient!!(rule, f, args...)[2][2:(K + 1)]
enz(f, args::Tuple{Vararg{Any, 5}}, _) = Enzyme.autodiff(Reverse, Const(f), Active, Active.(args[1:4])..., Const(args[5]))[1][1:4]
function enz(f, args::Tuple{Vararg{Any, 6}}, dx)
    dx .= 0
    r = Enzyme.autodiff(Reverse, Const(f), Active, Active.(args[1:4])..., Duplicated(args[5], dx), Const(args[6]))[1]
    return (r[1:4]..., dx)
end

for n in (10^3, 10^5)
    x = rand(n)
    idx = make_idx(n)
    dx = zero(x)
    println("\n## n = $n")
    println("| Computation | Storage of `w` | Forward | Zygote | Mooncake | Enzyme |")
    println("|---|---|---|---|---|---|")
    for (name, uses_x, build) in cases
        args = uses_x ? (0.3, 1.2, 0.7, 0.4, x, idx) : (0.3, 1.2, 0.7, 0.4, idx)
        K = length(args) - 1
        fd = build(dense_w)
        fp = build(pattern_w)
        @assert fd(args...) ≈ fp(args...)
        rd = Mooncake.build_rrule(fd, args...)
        rp = Mooncake.build_rrule(fp, args...)
        ref = moon(rd, fd, args, K)
        for (lab, f, r) in (("dense", fd, rd), ("PatternVector", fp, rp))
            ok(g) = grads_close(g, ref) ? "" : " (wrong)"
            okz, okm, oke = ok(zyg(f, args, K)), ok(moon(r, f, args, K)), ok(enz(f, args, dx))
            tf = @belapsed $f($args...)
            # time the bare AD calls; the wrappers above (tuple slicing) are only used to check the gradients
            eargs = (Active.(args[1:4])..., (uses_x ? (Duplicated(x, dx), Const(idx)) : (Const(idx),))...)
            tz = @belapsed Zygote.gradient($f, $args...)
            tm = @belapsed Mooncake.value_and_gradient!!($r, $f, $args...)
            te = uses_x ? @belapsed((fill!($dx, 0); Enzyme.autodiff(Reverse, Const($f), Active, $eargs...))) :
                @belapsed(Enzyme.autodiff(Reverse, Const($f), Active, $eargs...))
            println("| ", lab == "dense" ? name : "", " | $lab | $(fmt(tf)) | $(fmt(tz))$okz | $(fmt(tm))$okm | $(fmt(te))$oke |")
        end
    end
end

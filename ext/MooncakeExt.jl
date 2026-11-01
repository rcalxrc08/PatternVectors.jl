module MooncakeExt

using PatternVectors, Mooncake
using PatternVectors: materialize_mixture
using Mooncake: CoDual, NoRData, NoTangent, DefaultCtx, ReverseMode, primal, tangent, tangent_type, zero_tangent, zero_rdata, rdata, primal_to_tangent!!
using Mooncake.Nfwd: NDual
using Base: IEEEFloat
using Base.Broadcast: Broadcasted, instantiate

# Mooncake derives a structural tangent for PatternVector, Tangent{(n = NoTangent, pattern = Tangent{pattern fields})},
# so PatternVector-only code needs no rule. The ChainRulesCore rrules must NOT be imported with @from_rrule:
# their pullbacks expect an array-like cotangent and would make Mooncake materialize it.
#
# Rule for mixed broadcasts (PatternVectors with dense arrays, e.g. `w .* x`), hooked on `materialize_mixture`.
# The forward pass is the broadcast itself and stores nothing. The pullback makes one fused pass per differentiable
# argument, evaluating `f` again with that argument seeded as an NDual, so cheap functions such as `w .* x` cost
# little more than the broadcast. The passes are broadcasts and reductions: they run on any array type (e.g. CuArray).
# Unsupported signatures fall back to the rule derived by Mooncake.

# How an argument of the broadcast is differentiated. Foldable: constant at compile time in the pullback.
Base.@assume_effects :foldable function arg_role(::Type{A}) where {A}
    (tangent_type(A) == NoTangent || (A <: Base.RefValue && tangent_type(eltype(A)) == NoTangent)) && return :const
    (A <: AbstractArray{<:IEEEFloat} && tangent_type(A) == A) && return :dense # e.g. Array, CuArray
    A <: IEEEFloat && return :scalar
    (A <: PatternVector{<:IEEEFloat} && all(==(eltype(A)), fieldtypes(fieldtype(A, :pattern)))) && return :pattern
    return :unsupported
end

function Mooncake._is_primitive(::Type{DefaultCtx}, ::Type{<:ReverseMode}, ::Type{Sig}) where {Sig <: Tuple{typeof(materialize_mixture), Vararg}}
    isconcretetype(Sig) || return false
    _, S, F, Xs... = fieldtypes(Sig)
    (S <: Broadcast.AbstractArrayStyle && Base.issingletontype(F) && tangent_type(F) == NoTangent) || return false
    roles = map(arg_role, Xs)
    (:unsupported in roles || all(==(:const), roles)) && return false
    # one float type T: f keeps it on primals and propagates an NDual seeded in any differentiable argument
    elts = map(eltype, Xs)
    T = elts[findfirst(!=(:const), roles)]
    returns(E, R) = Core.Compiler.return_type(F.instance, Tuple{E...}) === R
    return returns(elts, T) && all(k -> roles[k] === :const || (elts[k] == T && returns(Base.setindex(elts, NDual{T, 1}, k), NDual{T, 1})), eachindex(Xs))
end

function Mooncake.rrule!!(::CoDual{typeof(materialize_mixture)}, style::CoDual, f::CoDual, args::Vararg{CoDual, M}) where {M}
    s, fn, xs = primal(style), primal(f), map(primal, args)
    y = materialize_mixture(s, fn, xs...)
    dy = zero(y)
    function materialize_mixture_pb(::NoRData)
        # elementwise cotangent of the k-th argument, dy .* ∂y/∂x_k: lazy, computed by the pass that consumes it
        Δ(k) = instantiate(Broadcasted(s, Partial(k, fn), (dy, xs...)))
        rdatas = map((dx, x, k) -> cotangent!(Val(arg_role(typeof(x))), dx, x, Δ(k)), map(tangent, args), xs, ntuple(Val, Val(M)))
        return (NoRData(), NoRData(), NoRData(), rdatas...)
    end
    return CoDual(y, dy), materialize_mixture_pb
end

# Δ * ∂f/∂x_k at one element: f evaluated with its k-th argument seeded as an NDual. Inlined: a call through varargs
# boxes the elements, so a fused expression too large for the inliner would allocate per element.
struct Partial{k, F}
    f::F
end
Partial(::Val{k}, f::F) where {k, F} = Partial{k, F}(f)
@inline (p::Partial{k})(Δ, vals::Vararg{Any, M}) where {k, M} = Δ * p.f(Base.setindex(vals, NDual(vals[k], (one(vals[k]),)), k)...).partials[1]

# Accumulates the cotangent Δ (lazy, shaped like the output) of argument x into its fdata dx and returns its rdata
cotangent!(::Val{:const}, dx, x, Δ) = zero_rdata(x)
cotangent!(::Val{:scalar}, dx, x, Δ) = sum(Δ)
cotangent!(::Val{:dense}, dx, x, Δ) = (dx .+= unbroadcast(x, Δ); NoRData())
function cotangent!(::Val{:pattern}, dx, x::PatternVector{T, P}, Δ) where {T, P}
    # The elements of a PatternVector whose fields are NDuals seeded per field carry their derivatives w.r.t. the
    # fields (for any pattern): one reduction gives all field cotangents. PatternVector is immutable, so they are
    # returned as rdata, converted by Mooncake from a PatternVector holding them.
    F, wrap = fieldcount(P), Base.typename(P).wrapper
    jac = PatternVector(x.n, wrap(ntuple(j -> NDual(getfield(x.pattern, j), ntuple(i -> T(i == j), Val(F))), Val(F))...))
    dfields = sum(instantiate(Broadcasted(Δ.style, *, (Δ, jac)))).partials
    return rdata(primal_to_tangent!!(zero_tangent(x), PatternVector(x.n, wrap(dfields...))))
end

# Sums Δ over the dimensions along which x was broadcast
function unbroadcast(x, Δ)
    size(x) == size(Δ) && return Δ
    dims = Tuple(d for (d, len) in enumerate(size(Δ)) if size(x, d) == 1 && len != 1)
    return reshape(sum(Base.materialize(Δ); dims), size(x))
end

end

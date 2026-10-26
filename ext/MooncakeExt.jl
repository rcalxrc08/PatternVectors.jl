module MooncakeExt

using PatternVectors, Mooncake
using PatternVectors: materialize_mixture
using Mooncake: CoDual, NoRData, NoTangent, Tangent, DefaultCtx, ReverseMode, primal, tangent, tangent_type, zero_rdata
using Mooncake.Nfwd: NDual
using Base: IEEEFloat
using Base.Broadcast: Broadcasted, DefaultArrayStyle, instantiate, preprocess, newindex

# Mooncake derives a structural tangent for PatternVector,
#   Tangent{@NamedTuple{n::NoTangent, pattern::tangent_type(P)}},
# so cotangents are accumulated directly onto the pattern fields and never onto a dense vector.
# The ChainRulesCore rrules must NOT be imported via @from_rrule: their pullbacks expect an
# array-like cotangent, which would force Mooncake to materialize the PatternVector cotangent.

# Fast reverse rule for mixed broadcasts (PatternVectors with dense arrays), e.g. `pv .* x`.
# Forward pass: plain broadcast, nothing is stored apart from the output.
# Reverse pass: a single fused loop re-evaluating `f` on NDuals with one direction per dense/scalar
# argument and one per pattern field; dense cotangents are accumulated in place, pattern and scalar
# cotangents in registers. Unsupported signatures fall back to the rule derived by Mooncake.

const MAX_DIRECTIONS = 16

struct ArgPlan
    role::Symbol # :const, :dense, :scalar or :pattern
    offset::Int  # directions of this argument are offset+1:offset+ndirs
    ndirs::Int
end

pattern_type(::Type{<:PatternVector{T, P}}) where {T, P} = P

function arg_role(::Type{A}) where {A}
    tangent_type(A) == NoTangent && return :const
    (A <: Base.RefValue && tangent_type(eltype(A)) == NoTangent) && return :const
    A <: Array{<:IEEEFloat} && return :dense
    A <: IEEEFloat && return :scalar
    (A <: PatternVector{<:IEEEFloat} && all(==(eltype(A)), fieldtypes(pattern_type(A)))) && return :pattern
    return :unsupported
end

# Element seen by the broadcast function for a non-differentiable argument
const_eltype(::Type{A}) where {A} = (A <: AbstractArray || A <: Base.RefValue) ? eltype(A) : A <: Number ? A : nothing

function mixture_plan(::Type{Xs}) where {Xs <: Tuple}
    plans = ArgPlan[]
    Ts = DataType[]
    K = 0
    for A in fieldtypes(Xs)
        role = arg_role(A)
        role === :unsupported && return nothing
        role === :const && const_eltype(A) === nothing && return nothing
        ndirs = role === :const ? 0 : role === :pattern ? fieldcount(pattern_type(A)) : 1
        role === :const || push!(Ts, eltype(A))
        push!(plans, ArgPlan(role, K, ndirs))
        K += ndirs
    end
    (K == 0 || K > MAX_DIRECTIONS || !allequal(Ts)) && return nothing
    return (plans = Tuple(plans), K = K, T = first(Ts))
end

function supports_fast_path(::Type{Sig}) where {Sig <: Tuple}
    isconcretetype(Sig) || return false
    S, F = fieldtype(Sig, 2), fieldtype(Sig, 3)
    S <: DefaultArrayStyle || return false
    (Base.issingletontype(F) && tangent_type(F) == NoTangent) || return false
    Xs = Tuple{fieldtypes(Sig)[4:end]...}
    plan = mixture_plan(Xs)
    plan === nothing && return false
    T, K = plan.T, plan.K
    primal_elts = Tuple{(p.role === :const ? const_eltype(A) : T for (p, A) in zip(plan.plans, fieldtypes(Xs)))...}
    dual_elts = Tuple{(p.role === :const ? const_eltype(A) : NDual{T, K} for (p, A) in zip(plan.plans, fieldtypes(Xs)))...}
    return Core.Compiler.return_type(F.instance, primal_elts) === T &&
        Core.Compiler.return_type(F.instance, dual_elts) === NDual{T, K}
end

function Mooncake._is_primitive(::Type{DefaultCtx}, ::Type{<:ReverseMode}, ::Type{Sig}) where {Sig <: Tuple{typeof(materialize_mixture), Vararg}}
    return supports_fast_path(Sig)
end

function Mooncake.rrule!!(::CoDual{typeof(materialize_mixture)}, style::CoDual, f::CoDual, args::Vararg{CoDual, M}) where {M}
    s, fn, xs = primal(style), primal(f), map(primal, args)
    y = materialize_mixture(s, fn, xs...)
    dy = zero(y)
    function materialize_mixture_pb(::NoRData)
        return (NoRData(), NoRData(), NoRData(), mixture_pullback(dy, fn, xs, map(tangent, args))...)
    end
    return CoDual(y, dy), materialize_mixture_pb
end

onehot(::Type{T}, ::Val{K}, k) where {T, K} = ntuple(i -> T(i == k), Val(K))
zero_acc(::Type{T}, ::Val{K}) where {T, K} = ntuple(_ -> zero(T), Val(K))
accumulate_partials(acc, Δ, partials) = map((a, q) -> muladd(Δ, q, a), acc, partials)

# Seeds direction k of an NDual{T,K} with the given primal value
struct Seed{T, K, k} end
(::Seed{T, K, k})(v) where {T, K, k} = NDual{T, K}(T(v), onehot(T, Val(K), k))

# Broadcast function seeding dense and scalar arguments elementwise before calling f
struct SeededFunction{F, W}
    f::F
    wraps::W
end
(s::SeededFunction)(vals...) = s.f(map((w, v) -> w(v), s.wraps, vals)...)

@generated function mixture_pullback(dy::AbstractArray{T, N}, f, xs::Xs, dxs) where {T, N, Xs}
    plan = mixture_plan(Xs)
    K = plan.K
    D = :(NDual{$T, $K})
    seeded_args = Any[]  # PatternVectors are seeded once, through their pattern fields
    wraps = Any[]        # dense and scalar elements are seeded inside the broadcast function
    dense_updates = Any[]
    results = Any[]
    for (i, p) in enumerate(plan.plans)
        o = p.offset
        if p.role === :pattern
            P = pattern_type(fieldtype(Xs, i))
            fields = [:($D(getfield(xs[$i].pattern, $j), onehot($T, Val($K), $(o + j)))) for j in 1:p.ndirs]
            push!(seeded_args, :(PatternVector(xs[$i].n, $(Base.typename(P).wrapper)($(fields...)))))
            push!(wraps, :identity)
            nt = :(NamedTuple{$(fieldnames(P))}(($([:(acc[$(o + j)]) for j in 1:p.ndirs]...),)))
            push!(results, :(Mooncake.rdata(Tangent((n = NoTangent(), pattern = Tangent($nt))))))
        elseif p.role === :const
            push!(seeded_args, :(xs[$i]))
            push!(wraps, :identity)
            push!(results, :(zero_rdata(xs[$i])))
        else
            push!(seeded_args, :(xs[$i]))
            push!(wraps, :(Seed{$T, $K, $(o + 1)}()))
            if p.role === :dense
                push!(dense_updates, :(dxs[$i][newindex(I, extruded[$i].keeps, extruded[$i].defaults)] += Δ * partials[$(o + 1)]))
                push!(results, :(NoRData()))
            else
                push!(results, :(acc[$(o + 1)]))
            end
        end
    end
    return quote
        g = SeededFunction(f, ($(wraps...),))
        bc = preprocess(dy, instantiate(Broadcasted(DefaultArrayStyle{$N}(), g, ($(seeded_args...),))))
        extruded = map(Base.Broadcast.extrude, xs)
        acc = zero_acc($T, Val($K))
        @inbounds for I in CartesianIndices(dy)
            Δ = dy[I]
            partials = bc[I].partials
            acc = accumulate_partials(acc, Δ, partials)
            $(dense_updates...)
        end
        return ($(results...),)
    end
end

end

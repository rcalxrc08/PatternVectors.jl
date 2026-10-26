module MooncakeExt

using PatternVectors, Mooncake
using PatternVectors: materialize_mixture, getindex_pattern
using Mooncake: CoDual, NoRData, NoTangent, Tangent, DefaultCtx, ReverseMode, primal, tangent, tangent_type, zero_rdata
using Mooncake.Nfwd: NDual, ndual_value
using Base: IEEEFloat
using Base.Broadcast: Broadcasted, DefaultArrayStyle, instantiate, preprocess, newindex

# Mooncake derives a structural tangent for PatternVector,
#   Tangent{@NamedTuple{n::NoTangent, pattern::tangent_type(P)}},
# so cotangents are accumulated directly onto the pattern fields and never onto a dense vector.
# The ChainRulesCore rrules must NOT be imported via @from_rrule: their pullbacks expect an
# array-like cotangent, which would force Mooncake to materialize the PatternVector cotangent.

# Fast reverse rule for mixed broadcasts (PatternVectors with dense arrays), e.g. `pv .* x`.
# `f` is evaluated on NDuals with one direction per differentiable argument (dense, scalar or PatternVector,
# each seeded elementwise), either again in the pullback or once in the forward pass (PARTIALS_STRATEGY).
# The reverse pass is a single fused loop: dense cotangents are accumulated in place, scalar cotangents in
# registers, and the elementwise cotangent of a PatternVector is mapped onto its pattern fields through the
# pattern Jacobian, obtained from `getindex_pattern` on a pattern with NDual fields (no evaluation of `f`).
# Unsupported signatures fall back to the rule derived by Mooncake.

const MAX_DIRECTIONS = 16

pattern_type(::Type{<:PatternVector{T, P}}) where {T, P} = P

# The plan depends on `tangent_type`, whose methods may come from extensions loaded later (e.g. CuArray in
# MooncakeCUDAExt). @generated bodies run in the world of their definition and would miss them, so the plan is
# computed by the callers (foldable: constant at compile time) and passed to the generated functions as Val.
Base.@assume_effects :foldable function arg_role(::Type{A}) where {A}
    tangent_type(A) == NoTangent && return :const
    (A <: Base.RefValue && tangent_type(eltype(A)) == NoTangent) && return :const
    # dense arrays whose fdata is an array of the same type, e.g. Array or CuArray (via MooncakeCUDAExt)
    (A <: AbstractArray{<:IEEEFloat} && tangent_type(A) == A) && return :dense
    A <: IEEEFloat && return :scalar
    (A <: PatternVector{<:IEEEFloat} && all(==(eltype(A)), fieldtypes(pattern_type(A)))) && return :pattern
    return :unsupported
end

# Element seen by the broadcast function for a non-differentiable argument
const_eltype(::Type{A}) where {A} = (A <: AbstractArray || A <: Base.RefValue) ? eltype(A) : A <: Number ? A : nothing

# roles[i] in (:const, :dense, :scalar, :pattern); directions[i] is the NDual direction of argument i (0 if :const)
Base.@assume_effects :foldable function mixture_plan(::Type{Xs}) where {Xs <: Tuple}
    roles = Symbol[]
    directions = Int[]
    Ts = DataType[]
    K = 0
    for A in fieldtypes(Xs)
        role = arg_role(A)
        role === :unsupported && return nothing
        role === :const && const_eltype(A) === nothing && return nothing
        role === :const || push!(Ts, eltype(A))
        K += role !== :const
        push!(roles, role)
        push!(directions, role === :const ? 0 : K)
    end
    (K == 0 || K > MAX_DIRECTIONS || !allequal(Ts)) && return nothing
    return (roles = Tuple(roles), directions = Tuple(directions), K = K, T = first(Ts))
end

function supports_fast_path(::Type{Sig}) where {Sig <: Tuple}
    isconcretetype(Sig) || return false
    S, F = fieldtype(Sig, 2), fieldtype(Sig, 3)
    S <: Broadcast.AbstractArrayStyle || return false
    (Base.issingletontype(F) && tangent_type(F) == NoTangent) || return false
    Xs = Tuple{fieldtypes(Sig)[4:end]...}
    plan = mixture_plan(Xs)
    plan === nothing && return false
    T, K = plan.T, plan.K
    primal_elts = Tuple{(role === :const ? const_eltype(A) : T for (role, A) in zip(plan.roles, fieldtypes(Xs)))...}
    dual_elts = Tuple{(role === :const ? const_eltype(A) : NDual{T, K} for (role, A) in zip(plan.roles, fieldtypes(Xs)))...}
    return Core.Compiler.return_type(F.instance, primal_elts) === T &&
        Core.Compiler.return_type(F.instance, dual_elts) === NDual{T, K}
end

function Mooncake._is_primitive(::Type{DefaultCtx}, ::Type{<:ReverseMode}, ::Type{Sig}) where {Sig <: Tuple{typeof(materialize_mixture), Vararg}}
    return supports_fast_path(Sig)
end

# Strategy for the partials used by the reverse pass:
# :store (default) evaluates f once on NDuals in the forward pass and keeps the K partials per element
#   (K * n extra floats); fastest unless f is very cheap.
# :recompute evaluates f on primals in the forward pass and again on NDuals in the pullback
#   (no memory besides the output); fastest for very cheap f, e.g. `pv .* x`.
const PARTIALS_STRATEGY = Ref(:store)

# The fused reverse loop indexes elements one by one: only for CPU Arrays. Other array types (e.g. CuArray)
# use the broadcast-based reverse pass, built from whole-array broadcasts and reductions.
Base.@assume_effects :foldable uses_fused_loop(::Type{S}, ::Type{Xs}) where {S, Xs} = S <: DefaultArrayStyle &&
    all(A -> arg_role(A) !== :dense || A <: Array, fieldtypes(Xs))

function Mooncake.rrule!!(::CoDual{typeof(materialize_mixture)}, style::CoDual, f::CoDual, args::Vararg{CoDual, M}) where {M}
    s, fn, xs = primal(style), primal(f), map(primal, args)
    full_plan = mixture_plan(typeof(xs))
    plan = Val((full_plan.roles, full_plan.directions, full_plan.K)) # isbits/Symbol tuples: a valid type parameter
    bc = seeded_broadcast(plan, s, fn, xs)
    if !uses_fused_loop(typeof(s), typeof(xs))
        yd = Base.materialize(bc) # f is evaluated once on NDuals, partials are kept (:store)
        y = map(ndual_value, yd)
        dy_gpu = zero(y)
        materialize_mixture_broadcast_pb(::NoRData) = (NoRData(), NoRData(), NoRData(), broadcast_pullback(plan, dy_gpu, yd, xs, map(tangent, args))...)
        return CoDual(y, dy_gpu), materialize_mixture_broadcast_pb
    end
    if PARTIALS_STRATEGY[] === :store
        y, partials = store_dual_broadcast(bc, dual_eltype(plan, xs))
    else
        y = materialize_mixture(s, fn, xs...)
        partials = nothing
    end
    dy = zero(y)
    function materialize_mixture_pb(::NoRData)
        src = partials === nothing ? preprocess(dy, bc) : partials
        return (NoRData(), NoRData(), NoRData(), mixture_pullback(plan, dy, src, xs, map(tangent, args))...)
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

# Common float type of the differentiable arguments (used inside generators, no tangent_type needed)
plan_eltype(roles, ::Type{Xs}) where {Xs} = eltype(fieldtype(Xs, findfirst(!=(:const), roles)))

@generated dual_eltype(::Val{plan}, xs::Xs) where {plan, Xs} = :(NDual{$(plan_eltype(plan[1], Xs)), $(plan[3])})

# Lazy broadcast of f over the arguments seeded with NDuals: element I holds y[I] and its K partials
@generated function seeded_broadcast(::Val{plan}, style, f, xs::Xs) where {plan, Xs}
    roles, directions, K = plan
    T = plan_eltype(roles, Xs)
    wraps = [role === :const ? :identity : :(Seed{$T, $K, $k}()) for (role, k) in zip(roles, directions)]
    return :(instantiate(Broadcasted(style, SeededFunction(f, ($(wraps...),)), xs)))
end

# Forward pass of the :store strategy: primal output and partials from a single dual evaluation
function store_dual_broadcast(bc::Broadcasted, ::Type{NDual{T, K}}) where {T, K}
    y = Array{T}(undef, map(length, axes(bc)))
    partials = Array{NTuple{K, T}}(undef, size(y))
    bcp = preprocess(y, bc)
    @inbounds for I in CartesianIndices(y)
        d = bcp[I]
        y[I] = d.value
        partials[I] = d.partials
    end
    return y, partials
end

Base.@propagate_inbounds partials_at(bc::Broadcasted, I) = bc[I].partials
Base.@propagate_inbounds partials_at(partials::Array, I) = partials[I]

# Pattern with NDual fields: getindex_pattern on it gives the derivatives of an element w.r.t. the pattern fields
@generated function seeded_pattern(pattern::P) where {P}
    T, F = fieldtype(P, 1), fieldcount(P)
    fields = [:(NDual{$T, $F}(getfield(pattern, $j), onehot($T, Val($F), $j))) for j in 1:F]
    return :($(Base.typename(P).wrapper)($(fields...)))
end

@generated function mixture_pullback(::Val{plan}, dy::AbstractArray{T}, src, xs::Xs, dxs) where {plan, T, Xs}
    roles, directions, K = plan
    setup = Any[]
    updates = Any[]
    results = Any[]
    for (i, (role, k)) in enumerate(zip(roles, directions))
        idx = :(newindex(I, extruded[$i].keeps, extruded[$i].defaults))
        if role === :pattern
            P = pattern_type(fieldtype(Xs, i))
            F = fieldcount(P)
            jac, pacc = Symbol(:jac, i), Symbol(:pacc, i)
            push!(setup, :($jac = seeded_pattern(xs[$i].pattern)), :($pacc = zero_acc($T, Val($F))))
            push!(updates, :($pacc = accumulate_partials($pacc, Δ * partials[$k], getindex_pattern($jac, $idx[1], xs[$i].n).partials)))
            nt = :(NamedTuple{$(fieldnames(P))}($pacc))
            push!(results, :(Mooncake.rdata(Tangent((n = NoTangent(), pattern = Tangent($nt))))))
        elseif role === :const
            push!(results, :(zero_rdata(xs[$i])))
        elseif role === :dense
            push!(updates, :(dxs[$i][$idx] += Δ * partials[$k]))
            push!(results, :(NoRData()))
        else
            push!(results, :(acc[$k]))
        end
    end
    return quote
        extruded = map(Base.Broadcast.extrude, xs)
        acc = zero_acc($T, Val($K))
        $(setup...)
        @inbounds for I in CartesianIndices(dy)
            Δ = dy[I]
            partials = partials_at(src, I)
            acc = accumulate_partials(acc, Δ, partials)
            $(updates...)
        end
        return ($(results...),)
    end
end

# Broadcast-based reverse pass (any array type supporting broadcast and reductions, e.g. CuArray).
# Cotangent of argument k, elementwise: dy .* ∂y/∂x_k, read from the stored NDuals.
struct PartialTimes{k} end
(::PartialTimes{k})(Δ, d) where {k} = Δ * d.partials[k]

# Sum Δ over the dimensions broadcast to reach its shape from an argument of size sz
function reduce_to(Δ::AbstractArray, sz::Tuple)
    dims = Tuple(d for d in 1:ndims(Δ) if d > length(sz) || (sz[d] == 1 && size(Δ, d) != 1))
    return isempty(dims) ? Δ : reshape(sum(Δ; dims = dims), sz)
end

pattern_rdata(dp::P) where {P} = Mooncake.rdata(Tangent((n = NoTangent(), pattern = Tangent(NamedTuple{fieldnames(P)}(ntuple(j -> getfield(dp, j), fieldcount(P)))))))

@generated function broadcast_pullback(::Val{plan}, dy, yd, xs::Xs, dxs) where {plan, Xs}
    roles, directions = plan
    results = Any[]
    for (i, (role, k)) in enumerate(zip(roles, directions))
        Δk = :(Broadcast.broadcasted(PartialTimes{$k}(), dy, yd))
        if role === :const
            push!(results, :(zero_rdata(xs[$i])))
        elseif role === :scalar
            push!(results, :(sum(Base.materialize($Δk))))
        elseif role === :dense
            push!(results, quote
                if size(dxs[$i]) == size(dy)
                    dxs[$i] .+= $Δk
                else
                    dxs[$i] .+= reduce_to(Base.materialize($Δk), size(dxs[$i]))
                end
                NoRData()
            end)
        else
            P = pattern_type(fieldtype(Xs, i))
            push!(results, :(pattern_rdata(PatternVectors.pattern_to_vector_pullback($P, reduce_to(Base.materialize($Δk), size(xs[$i])), xs[$i].n))))
        end
    end
    return :(($(results...),))
end

end

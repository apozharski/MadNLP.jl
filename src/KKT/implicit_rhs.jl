"""
    ImplicitKKTVector{T, VT<:AbstractVector{T}} <: AbstractKKTVector{T, VT}

Full KKT vector ``(x, s, v, y, z)``, associated to a [`AbstractUnreducedKKTSystem`](@ref).

"""
struct ImplicitKKTVector{T,VT<:AbstractVector{T}} <: AbstractKKTVector{T,VT}
    values::VT
    v::VT  # unsafe view
    y::VT # unsafe view
    xp::VT # unsafe view
end

function ImplicitKKTVector(::Type{VT}, n::Int, m::Int, nlb::Int, nub::Int, ind_lb, ind_ub) where VT
    n_B = nlb + nub
    values = VT(undef, n+n_B+m)
    xp = _madnlp_unsafe_wrap(values, n) # (x,s)
    v = _madnlp_unsafe_wrap(values, n_B, n+1) # (v)
    y = _madnlp_unsafe_wrap(values, m, n+m+1) # (y)
    return ImplicitKKTVector(
        values,
        v,
        y,
        xp
    )
end

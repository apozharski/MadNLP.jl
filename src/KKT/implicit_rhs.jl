"""
    ImplicitKKTVector{T, VT<:AbstractVector{T}} <: AbstractKKTVector{T, VT}

Full KKT vector ``(x, s, v, y, z)``, associated to a [`AbstractUnreducedKKTSystem`](@ref).

"""
struct ImplicitKKTVector{T,VT} <: AbstractKKTVector{T,VT}
    values::VT
    x::VT  # unsafe view
    s::VT  # unsafe view
    v::VT  # unsafe view
    y::VT # unsafe view
    xp::VT # unsafe view
end

function ImplicitKKTVector(::Type{VT}, n, n_s, m, n_B)
    values = VT(undef, n+n_B+m)
    x = _madnlp_unsafe_wrap(values, n) # (x)
    s = _madnlp_unsafe_wrap(values, n_s, n+1) # (x)
    xp = _madnlp_unsafe_wrap(values, n+n_s) # (x,s)
    v = _madnlp_unsafe_wrap(values, n_B, n+n_s+1) # (v)
    y = _madnlp_unsafe_wrap(values, m, n+n_s+m+1) # (y)
end

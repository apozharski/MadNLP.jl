"""
    ImplicitKKTSystem{T, VT, MT, QN}

Implicit KKT system from 'Implicit Primal-Dual Interior-Point Methods for Quadratic Programming'

"""

struct ImplicitKKTSystem{T, VT, MT, QN, LS, VI, VI32} <: AbstractKKTSystem{T, VT, MT, QN}
    # Hessian
    hess::VT
    hess_raw::SparseMatrixCOO{T, Int32, VT, VI32}
    hess_com::MT
    hess_csc_map::Union{Nothing, VI}

    # Jacobian
    jac::VT
    jac_raw::SparseMatrixCOO{T, Int32, VT, VI32}
    jac_com::MT
    jac_csc_map::Union{Nothing, VI}

    # Augmented system
    aug_raw::SparseMatrixCOO{T, Int32, VT, VI32}
    aug_com::MT
    aug_csc_map::Union{Nothing, VI}

    # Diags
    reg::VT
    pr_diag::VT
    du_diag::VT
    l_diag::VT
    u_diag::VT
    l_lower::VT
    u_lower::VT

    # LinearSolver
    linear_solver::LS
    quasi_newton::QN

    w_buffer::VT

    # Info
    n_var::Int
    n_slack::Int
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI
end


function create_kkt_system(
    ::Type{ImplicitKKTSystem},
    cb::SparseCallback{T,VT},
    linear_solver::Type;
    opt_linear_solver=default_options(linear_solver),
    hessian_approximation=ExactHessian,
    qn_options=QuasiNewtonOptions(),
) where {T,VT}
    n_slack = length(cb.ind_ineq)
    ind_ineq = cb.ind_ineq
    ind_lb = cb.ind_lb
    ind_ub = cb.ind_ub

    n = cb.nvar
    m = cb.ncon

    quasi_newton = create_quasi_newton(hessian_approximation, cb, n; options=qn_options)

    jac_sparsity_I = create_array(cb, Int32, cb.nnzj)
    jac_sparsity_J = create_array(cb, Int32, cb.nnzj)
    _jac_sparsity_wrapper!(cb,jac_sparsity_I, jac_sparsity_J)

    hess_sparsity_I, hess_sparsity_J = build_hessian_structure(cb, hessian_approximation)
    force_lower_triangular!(hess_sparsity_I,hess_sparsity_J)

    n_jac = length(jac_sparsity_I)
    n_hess = length(hess_sparsity_I)
    nlb = length(cb.ind_lb)
    nub = length(cb.ind_ub)
    n_tot = n + n_slack
    n_B = nlb+nub

    # (n+n_slack) + n_B + m
    aug_vec_length = n_tot+n_B+m
    # Hesssian + Jacobian + A + ATA + B + pr_diag + du_diag
    aug_mat_length = n_hess+n_jac+(n_B)+(n_B)+(n_B) + n_tot + (m+n_B)

    I = create_array(cb, Int32, aug_mat_length)
    J = create_array(cb, Int32, aug_mat_length)
    V = VT(undef, aug_mat_length)
    fill!(V, 0.0)  # Need to initiate V to avoid NaN

    # Hessian
    I[1:n_hess] .= hess_sparsity_I
    J[1:n_hess] .= hess_sparsity_J
    # ATA (1,1) block
    I[n_hess+1:n_hess+nlb] .= cb.ind_lb
    J[n_hess+1:n_hess+nlb] .= cb.ind_lb
    I[n_hess+nlb+1:n_hess+nlb+nub] .= cb.ind_ub
    J[n_hess+nlb+1:n_hess+nlb+nub] .= cb.ind_ub
    # A
    I[n_hess+n_B+1:n_hess+n_B+nlb] .= n_tot .+ (1:nlb)
    J[n_hess+n_B+1:n_hess+n_B+nlb] .= cb.ind_lb
    I[n_hess+n_B+nlb+1:n_hess+n_B+nlb+nub] .= n_tot+nlb .+ (1:nub)
    J[n_hess+n_B+nlb+1:n_hess+n_B+nlb+nub] .= cb.ind_ub
    # B
    I[n_hess+2*n_B+1:n_hess+2*n_B+nlb] .= n_tot .+ (1:nlb)
    J[n_hess+2*n_B+1:n_hess+2*n_B+nlb] .= n_tot .+ (1:nlb)
    I[n_hess+2*n_B+nlb+1:n_hess+2*n_B+nlb+nub] .= n_tot+nlb .+ (1:nub)
    J[n_hess+2*n_B+nlb+1:n_hess+2*n_B+nlb+nub] .= n_tot+nlb .+ (1:nub)
    # Jacobian
    I[n_hess+3*n_B+1:n_hess+3*n_B+n_jac] .= (n_tot+n_B) .+ jac_sparsity_I
    J[n_hess+3*n_B+1:n_hess+3*n_B+n_jac] .= jac_sparsity_J
    #pr_diag/du_diag
    I[n_hess+3*n_B+n_jac+1:n_hess+3*n_B+n_jac+n_tot+m+n_B] .= 1:(n_tot+m+n_B)
    J[n_hess+3*n_B+n_jac+1:n_hess+3*n_B+n_jac+n_tot+m+n_B] .= 1:(n_tot+m+n_B)

    pr_diag = _madnlp_unsafe_wrap(V, n_tot, n_hess+3*n_B+n_jac+1)
    du_diag = _madnlp_unsafe_wrap(V, m+n_B, n_hess+3*n_B+n_jac+n_tot+1)

    hess = _madnlp_unsafe_wrap(V, n_hess)
    jac = _madnlp_unsafe_wrap(V, n_jac+n_slack, n_hess+3*n_B+1)

    aug_raw = SparseMatrixCOO(aug_vec_length,aug_vec_length,I,J,V)
    jac_raw = SparseMatrixCOO(
        m, n_tot,
        Int32[jac_sparsity_I; ind_ineq],
        Int32[jac_sparsity_J; n+1:n+n_slack],
        jac,
    )

    hess_raw = SparseMatrixCOO(
        n_tot, n_tot,
        hess_sparsity_I,
        hess_sparsity_J,
        hess,
    )

    aug_com, aug_csc_map = coo_to_csc(aug_raw)
    jac_com, jac_csc_map = coo_to_csc(jac_raw)
    hess_com, hess_csc_map = coo_to_csc(hess_raw)

    reg = VT(undef, n_tot)
    l_diag = VT(undef, nlb)
    u_diag = VT(undef, nub)
    l_lower = VT(undef, nlb)
    u_lower = VT(undef, nub)

    _linear_solver = linear_solver(
        aug_com; opt = opt_linear_solver
    )

    w_buffer = ImplicitKKTVector(n, n_slack, m, n_B)

    return ImplicitKKTSystem(
        hess, hess_raw, hess_com, hess_csc_map,
        jac, jac_raw, jac_com, jac_csc_map,
        aug_raw, aug_com, aug_csc_map,
        reg, pr_diag, du_diag,
        l_diag, u_diag,
        l_lower,u_lower,
        _linear_solver,
        quasi_newton,
        w_buffer,
        n,
        n_slack,
        ind_ineq,
        ind_lb,
        ind_ub,
    )
end

num_variables(kkt::ImplicitKKTSystem) = kkt.n_var

function solve_kkt!(kkt::ImplicitKKTSystem, w::AbstractKKTVector{T,VT})
    xp = kkt.w_buffer.xp
    v = kkt.w_buffer.v
    y = kkt.w_buffer.y

    w_xp = primal(w)

    xp .= w_x

end

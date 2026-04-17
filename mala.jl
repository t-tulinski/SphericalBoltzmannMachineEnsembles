function mala!(
    sbm::SphericalBoltzmannMachine,
    evecs::AbstractMatrix,
    evals::AbstractVector,
    K::Integer,
    β::Real,
    γ::Real,
    dt::Real,
    tracker::AcceptanceTracker,
)
    J = sbm.J
    μ = sbm.μ

    grad_J = ∂LL(J, μ, evecs, evals, K, γ)

    N = size(J, 1)
    η = GOE_mat(N)
    ΔJ = Euler_Maruyama_step(Matrix(grad_J), Matrix(η); dt=dt, β=β)

    J_prop = Symmetric(Matrix(J) + ΔJ)
    μ_prop = optimal_μ(eigvals(J_prop))
    grad_J_prop = ∂LL(J_prop, μ_prop, evecs, evals, K, γ)

    logp_curr = β * LL(J, μ, evecs, evals, K, γ)
    logp_prop = β * LL(J_prop, μ_prop, evecs, evals, K, γ)

    J_curr_mat = Matrix(J)
    J_prop_mat = Matrix(J_prop)
    grad_curr_mat = Matrix(grad_J)
    grad_prop_mat = Matrix(grad_J_prop)

    logq_forward = -sum(abs2, J_prop_mat - J_curr_mat - dt * grad_curr_mat) / (4 * dt / β)
    logq_backward = -sum(abs2, J_curr_mat - J_prop_mat - dt * grad_prop_mat) / (4 * dt / β)

    log_alpha = logp_prop - logp_curr + logq_backward - logq_forward

    if log(rand()) < min(0.0, log_alpha)
        sbm.J = J_prop
        sbm.μ = μ_prop
        tracker.accepted += 1
    end

    tracker.total += 1
    return acceptance_rate(tracker)
end
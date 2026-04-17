function initialize_sbm(N::Int, σ::Real)
    J = GOE_mat(N) .* σ 
    λs = eigvals(J)
    μ = optimal_μ(λs)
    return SphericalBoltzmannMachine(Symmetric(J), μ)
end
function ∂E(evecs::AbstractMatrix, evals::AbstractVector, K::Integer)
    N = size(evecs, 1)
    size(evecs, 2) == N || throw(DimensionMismatch("evecs must be square."))
    length(evals) == N || throw(DimensionMismatch("length(evals) must equal size(evecs, 1)."))
    K > 0 || throw(ArgumentError("K must be positive."))
    return Symmetric(0.5 * evecs' * Diagonal(evals) * evecs)
end

function model_covariance(J::Symmetric, μ::Real; check::Bool=false, ε::Real=1e-6)
    F = eigen(J)
    inv_diag = Diagonal(inv.(μ .- F.values))
    C_model = Symmetric(F.vectors * inv_diag * F.vectors')
    if check
        n = size(J, 1)
        I_n = Matrix{Float64}(I, n, n)
        residual = opnorm(Matrix(C_model) * Matrix(μ * I - J) - I_n, Inf)
        residual < ε || throw(ErrorException("Inversion of model precision matrix failed."))
    end
    return C_model
end

function ∂logZ(J::Symmetric, μ::Real, K::Integer; check::Bool=false, ε::Real=1e-6)
    return Symmetric(0.5 * K * model_covariance(J, μ; check=check, ε=ε))
end

function ∂L2(J::Symmetric, γ::Real)
    N = size(J, 1)
    return Symmetric(0.5 * N * γ * J)
end

function ∂LL(J::Symmetric, μ::Real, evecs::AbstractMatrix, evals::AbstractVector, K::Integer, γ::Real)
    return Symmetric(
        ∂E(evecs, evals, K) -
        ∂L2(J, γ) -
        ∂logZ(J, μ, K)
    )
end
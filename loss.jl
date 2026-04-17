function data_covar(evecs::AbstractMatrix, evals::AbstractVector, K::Integer)
    N = size(evecs, 1)
    size(evecs, 2) == N || throw(DimensionMismatch("evecs must be square."))
    length(evals) == N || throw(DimensionMismatch("length(evals) must equal size(evecs, 1)."))
    K > 0 || throw(ArgumentError("K must be positive."))
    return evecs' * Diagonal(evals) * evecs / K
end

function E(J::Symmetric, evecs::AbstractMatrix, evals::AbstractVector, K::Integer)
    C = data_covar(evecs, evals, K)
    return 0.5 * K * tr(J * C)
end

function logZ(J::Symmetric, μ::Real, K::Integer, N::Integer)
    return -0.5 * K * logdet(μ * I - J) + 0.5 * K * N * μ
end

function L2(J::Symmetric, γ::Real)
    N = size(J, 1)
    return 0.25 * N * γ * sum(abs2, J)
end

function LL(J::Symmetric, μ::Real, evecs::AbstractMatrix, evals::AbstractVector, K::Integer, γ::Real)
    N = size(evecs, 1)
    return E(J, evecs, evals, K) - L2(J, γ) - logZ(J, μ, K, N)
end
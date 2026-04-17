using LinearAlgebra
using Serialization
using ToeplitzMatrices: Circulant

# bump-like data
function receptive_fields_covariance_circular(
    N::Int; tau::Real = 0.2, center::Real = 0.5
)
    N > 1 || throw(ArgumentError("N must be > 1"))
    tau > 0 || throw(ArgumentError("tau must be > 0"))
    0 <= center <= 1 || throw(ArgumentError("center must be in [0, 1]"))

    n = 0:N-1
    c = center * N
    d = @. min(abs(n - c), N - abs(n - c))
    sigma_samples = sqrt(tau) * (N - 1)

    v = @. exp(-((d / sigma_samples)^2))
    v ./= sum(v)
    v .-= inv(N)

    return Circulant(v)
end

function sample_receptive_fields(
    K::Int,
    Sigma::AbstractMatrix;
    random::Bool = true,
    replace::Bool = false,
)
    N = size(Sigma, 1)
    size(Sigma, 1) == size(Sigma, 2) || throw(ArgumentError("Sigma must be square"))

    if random
        if !replace && K > N
            throw(ArgumentError("K must be <= N when sampling without replacement"))
        end
        indices = replace ? rand(1:N, K) : randperm(N)[1:K]
    else
        indices = 1 .+ floor.(Int, ((0:K-1) .+ 0.5) .* N ./ K)
    end

    return Sigma[indices, :]
end

function generate_receptive_field_data(;
    N::Int = 1000,
    K::Int = 800,
    tau::Real = 0.1,
    center::Real = 0.5,
    sigma_noise::Real = 1e-6,
)
    Sigma = receptive_fields_covariance_circular(N; tau = tau, center = center)

    X = permutedims(sample_receptive_fields(K, Sigma; random = true, replace = true))
    X .+= sigma_noise .* randn(size(X))
    norms = sqrt.(sum(abs2, X; dims = 1))
    X ./= norms
    X .*= sqrt(N)

    C = Symmetric(X * X' / K)
    F = eigen(C)
    eigvals = reverse(F.values .* (K / N))
    eigvecs = reverse(F.vectors .* sqrt(N), dims = 2)

    return (; C, eigvals, eigvecs)
end
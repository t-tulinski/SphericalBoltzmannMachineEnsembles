function optimal_μ(λ::AbstractVector{<:Real}; ε::Union{Nothing,Real}=nothing)
    isempty(λ) && throw(ArgumentError("Eigenvalue vector must be non-empty."))

    λmax = maximum(λ)
    eps = isnothing(ε) ? 1e-6 / length(λ) : float(ε)

    G(μ) = 1 - mean(inv.(μ .- λ))

    a = λmax + eps
    b = a + max(1.0, abs(float(λmax)))

    # until we get a valid finite bisection interval G(a) < 0 < G(b)
    while G(b) <= 0
        b = λmax + 2 * (b - λmax)
        isfinite(b) || throw(ErrorException("Could not bracket the root for μ."))
    end

    return find_zero(G, (a, b), Bisection())
end

function constraint_status(λ::AbstractVector{<:Real}, μ::Real; ε::Real=1e-6)
    λmax = maximum(λ)

    saddle_point_error = abs(1 - mean(inv.(μ .- λ)))
    saddle_point_ok = saddle_point_error < ε

    eigenvalue_gap = μ - λmax
    eigenvalue_ok = eigenvalue_gap >= -ε

    return (
        saddle_point_ok = saddle_point_ok,
        saddle_point_error = saddle_point_error,
        eigenvalue_ok = eigenvalue_ok,
        eigenvalue_gap = eigenvalue_gap,
    )
end
function Euler_Maruyama_step(
    gradient::AbstractMatrix,
    noise::AbstractMatrix;
    dt::Real,
    β::Real,
)
    dt > 0 || throw(ArgumentError("dt must be positive."))
    β > 0 || throw(ArgumentError("β must be positive."))
    size(gradient) == size(noise) || throw(DimensionMismatch("gradient and noise must have the same size."))

    return dt .* gradient .+ sqrt(2 * dt / β) .* noise
end
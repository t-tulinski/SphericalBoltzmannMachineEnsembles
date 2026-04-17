function GOE_mat(N::Int)
    N > 0 || throw(ArgumentError("N must be positive."))
    W = randn(N, N)
    return Symmetric((W + W') / sqrt(2 * N))
end
mutable struct SphericalBoltzmannMachine
    J::Symmetric{Float64, Matrix{Float64}} 
    μ::Real          
    λs::Union{Vector{Float64}, Nothing}  
    function SphericalBoltzmannMachine(J, μ)
        new(J, μ, nothing) 
    end
end
# ------------------------------------------------------------------
# Monte Carlo helpers
# ------------------------------------------------------------------

mutable struct AcceptanceTracker
    accepted::Int
    total::Int
end

acceptance_rate(tracker::AcceptanceTracker) = 
    tracker.total == 0 ? 0.0 : tracker.accepted / tracker.total

function initialize_trace(; save_J::Bool=false)
    return (
        J_samples = save_J ? Matrix{Float64}[] : nothing,
        μ_samples = Float64[],
        λ1_samples = Float64[],
        q1_samples = Float64[],
        u1_samples = Float64[],
    )
end

function record_sample!(trace, sbm::SphericalBoltzmannMachine, evecs::AbstractMatrix)
    F = eigen(sbm.J)
    λ = F.values
    V = F.vectors
    N = size(evecs, 2)
    v1 = V[:, end]
    push!(trace.μ_samples, sbm.μ)
    push!(trace.λ1_samples, λ[end])
    push!(trace.q1_samples, 1 / (N * (sbm.μ - λ[end])))
    push!(trace.u1_samples, (dot(v1, evecs[1, :]) / sqrt(N))^2)
    if trace.J_samples !== nothing
        push!(trace.J_samples, copy(Matrix(sbm.J)))
    end
    return nothing
end

function latest_trace_summary(trace)
    isempty(trace.μ_samples) && return "no samples collected yet"
    return join([
        "μ = $(trace.μ_samples[end])",
        "λ1 = $(trace.λ1_samples[end])",
        "q1 = $(trace.q1_samples[end])",
        "u1 = $(trace.u1_samples[end])",
    ], ", ")
end

# ------------------------------------------------------------------
# Monte Carlo sampling of the model ensemble
# ------------------------------------------------------------------
function sample_sbm!(
    sbm::SphericalBoltzmannMachine,
    evecs::AbstractMatrix,
    evals::AbstractVector,
    K::Integer,
    β::Real,
    γ::Real,
    dt::Real,
    maxiter::Integer,
    sample_every::Integer;
    log_every::Integer=1_000,
    save_J::Bool=false,
)

    maxiter > 0 || throw(ArgumentError("maxiter must be positive."))
    sample_every > 0 || throw(ArgumentError("sample_every must be positive."))
    log_every > 0 || throw(ArgumentError("log_every must be positive."))

    tracker = AcceptanceTracker(0, 0)
    trace = initialize_trace(save_J=save_J)

    mkpath("logs")
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    log_file = "logs/$(timestamp)-sbm-sampling-beta=$(β)-gamma=$(γ)-dt=$(dt).log"

    last_log_time = time()
    elapsed_time = 0.0

    for iter in 1:maxiter
        acc_rate = mala!(sbm, evecs, evals, K, β, γ, dt, tracker)
        should_record =
            tracker.accepted > length(trace.λ1_samples) &&
            (iter % sample_every == 0)

        if should_record
            record_sample!(trace, sbm, evecs)
        end

        status = constraint_status(eigvals(sbm.J), sbm.μ)

        status.saddle_point_ok ||
            error("Saddle-point condition violated at step $iter: error = $(status.saddle_point_error)")

        status.eigenvalue_ok ||
            error("Eigenvalue condition violated at step $iter: gap = $(status.eigenvalue_gap)")

        if iter % log_every == 0
            current_time = time()
            elapsed_time += current_time - last_log_time
            last_log_time = current_time

            ll_value = LL(sbm.J, sbm.μ, evecs, evals, K, γ)

            log_message =
                "Iteration $iter, " *
                "t = $(round(elapsed_time, digits=2)) s, " *
                "AR = $acc_rate, " *
                "collected $(length(trace.λ1_samples)) samples, " *
                "log-likelihood = $ll_value, " *
                latest_trace_summary(trace)

            @info log_message

            open(log_file, "a") do io
                println(io, "[$(now())] $log_message")
            end
        end
    end

    return save_J ?
        (
            J_samples = trace.J_samples,
            μ_samples = trace.μ_samples,
            λ1_samples = trace.λ1_samples,
            q1_samples = trace.q1_samples,
            u1_samples = trace.u1_samples,
        ) :
        (
            J = sbm.J,
            μ_samples = trace.μ_samples,
            λ1_samples = trace.λ1_samples,
            q1_samples = trace.q1_samples,
            u1_samples = trace.u1_samples,
        )
end
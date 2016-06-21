using Distributions

type Kalman
    xs::Array{Float64,1}
    ps::Array{Float64,2}
    K::Int64
end

function init_kalman(K::Int64 = 2)
    return Kalman([10.0 for i in 1:K], eye(K) * 100.0, K)
end

# TODO two kinds of q
function filter_predict(kalman::Kalman, ML_q::Float64, q_inference_type::Int64, evolvability_type::AbstractString, population_index::Int64)
    q = 1.0
    if q_inference_type != 0
        q = ML_q
    end
    #kalman.ps += eye(kalman.K) * q
    kalman.ps[population_index, population_index] += q
end

# TODO something with n, should be an array
function filter_update(kalman::Kalman, obs::Array{Float64,1}, sample_sizes::Array{Int64, 1}, evolvability_type::AbstractString, population_index::Int64)
    for (ob, sample_size) in zip(obs, sample_sizes)
        # The variance of the observation, depending on the observation type
        if evolvability_type == "variance"
            r = 2(kalman.xs[population_index]^2) / Float64(sample_size)  # Observation variance
        elseif evolvability_type == "std"
            r = 1.0/(2.0 * (sample_size-2.0+1.0/(pi-2.0))) * kalman.xs[population_index]^2
        else # If we observe the maximum
            r = (0.125 + 1.29 * Float64(sample_size-1)^(-0.73))^2 * kalman.xs[population_index]^2
        end

        y = ob - kalman.xs[population_index]
        s = kalman.ps[population_index, population_index] + r
        k = kalman.ps[:, population_index] * 1/s

        # TODO check
        kalman.xs += k * y
        if evolvability_type != "maximum"
            kalman.xs[kalman.xs .< 0] = 0.0001
        end
        e = eye(kalman.K)
        e[:, population_index] -= k
        kalman.ps = e * kalman.ps
    end
end

function filter_expected_values(kalman::Kalman)
    return kalman.xs
end

function filter_probabilities(kalman::Kalman)
    # Calculate probability that each is maximal
    dist = MvNormal(kalman.xs, kalman.ps + 0.01 * eye(kalman.K))
    samples = rand(dist, 1000)
    maxes = mod(squeeze(findmax(samples, 1)[2], 1) - 1, kalman.K) + 1
    probs = hist(maxes, 0.5:kalman.K+0.5)[2] / length(maxes)
end

function filter_duplicate(kalman::Kalman, index::Int64)
    kalman.xs = repmat([kalman.xs[index]], kalman.K)
    kalman.ps = ones(kalman.K, kalman.K) * kalman.ps[index, index]
end

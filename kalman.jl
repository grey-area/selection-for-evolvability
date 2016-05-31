type Kalman
    xs::Array{Float64,1}
    ps::Array{Float64,2}
end

function init_kalman()
    return Kalman([10.0, 10.0], [100.0 0.0; 0.0 100.0])
end

function kalman_pred(kalman::Kalman, ML_q::Float64, q_inference_type::Int64)
    q = 1.0
    if q_inference_type == 1
        q = ML_q
    end
    kalman.ps += eye(2) * q
end

function kalman_update(kalman::Kalman, obs::Array{Float64,2}, n::Int64, evolvability_type::ASCIIString)

    local r::Array{Float64,1}

    # The variance of the observation, depending on the observation type
    if evolvability_type == "variance"
        r = 2(kalman.xs.^2) / Float64(n)  # Observation variance
    elseif evolvability_type == "std"
        r = 1.0/(2.0 * (n-2.0+1.0/(pi-2.0))) * kalman.xs.^2
    else # If we observe the maximum
        r = (0.125 + 1.29 * Float64(n-1)^(-0.73))^2 * kalman.xs.^2
    end

    for row in 1:size(obs)[2]
        y = obs[:, row] - kalman.xs
        s = kalman.ps + diagm(r)
        k = kalman.ps * inv(s)
        kalman.xs += k * y
        if evolvability_type != "maximum"
            kalman.xs[kalman.xs .< 0] = 0.0001
        end
        kalman.ps = (eye(2) - k) * kalman.ps
    end
end

function kalman_duplicate(kalman::Kalman, index::Int64)
    kalman.xs = repmat([kalman.xs[index]], 2)
    kalman.ps = ones(2,2) * kalman.ps[index, index]
end

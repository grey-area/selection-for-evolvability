type Kalman
    xs::Array{Float64,1}
    ps::Array{Float64,2}
end

function init_kalman()
    return Kalman([10.0, 10.0], [100.0 0.0; 0.0 100.0])
end

function kalman_obs(kalman::Kalman, obs::Array{Float64,1}, n::Int64, evolvability_type::ASCIIString)

    local r::Array{Float64,1}

    # The variance of the observation, depending on the observation type
    if evolvability_type == "variance"
        r = 2(kalman.xs.^2) / Float64(n)  # Observation variance
    elseif evolvability_type == "std"
        r = 1.0/(2.0 * (n-2.0+1.0/(pi-2.0))) * kalman.xs.^2
    else # If we observe the maximum
        r = (0.125 + 1.29 * Float64(n-1)^(-0.73))^2 * kalman.xs.^2
    end

    q = 1.0
    p1 = kalman.ps + eye(2) * q
    y = obs - kalman.xs
    s = p1 + diagm(r)
    k = p1 * inv(s)
    kalman.xs += k * y
    kalman.xs[kalman.xs .< 0] = 0.0001
    kalman.ps = (eye(2) - k) * p1
end

function kalman_duplicate(kalman::Kalman, index::Int64)
    kalman.xs = repmat([kalman.xs[index]], 2)
    kalman.ps = ones(2,2) * kalman.ps[index, index]
end

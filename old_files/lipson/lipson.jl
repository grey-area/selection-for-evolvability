using StatsBase

function get_targets()
    left_objects = [1 1 1 1; 1 1 1 0; 1 0 1 0; 1 0 1 1; 1 0 0 0; 1 1 0 1; 0 0 1 0; 0 1 1 1]
    right_objects = [1 1 1 1; 1 1 1 0; 0 1 0 1; 1 0 1 1; 0 0 0 1; 1 1 0 1; 0 1 0 0; 0 1 1 1]

    # TODO tidy up
    inputs = [[parse(Int64, c) for c in bits(i)[end-7:end]] for i in 0:255]
    and_targets = falses(256)
    or_targets = falses(256)
    for (input_i, input) in enumerate(inputs)
        left = false
        right = false
        for j in 1:8
            if input[1:4] == squeeze(left_objects[j, :], 1)
                left = true
            end
            if input[5:8] == squeeze(right_objects[j, :], 1)
                right = true
            end
        end
        and_targets[input_i] = left && right
        or_targets[input_i] = left || right
    end
    targets = [and_targets or_targets]

    return inputs, targets
end

type Network
    w1::Array{Int64, 2}
    w2::Array{Int64, 2}
    w3::Array{Int64, 2}
    w4::Array{Int64, 2}
end

import Base.copy
Base.copy(network::Network) = Network(copy(network.w1), copy(network.w2), copy(network.w3), copy(network.w4))

function evaluate(network::Network, input::Array{Int64, 1})
    lambda = 20
    output = copy(input)

    for layer in Any[network.w1, network.w2, network.w3, network.w4]
        push!(output, 1.0)
        output = tanh(lambda * layer * output)
    end

    return output[1] > 0
end

function random_network()
    return Network(rand(-2:2, 8, 9), rand(-2:2, 4, 9), rand(-2:2, 2, 5), rand(-2:2, 1, 3))
end

function network_accuracy(network::Network, inputs, targets, task::Int64)
    return sum(Bool[evaluate(network, input) == targets[input_i, task] for (input_i, input) in enumerate(inputs)])
end

function inc_dec(layer::Array{Int64, 2}, i::Int64, j::Int64)
    if rand() < 0.01

        v = layer[i, j]
        if rand() > 0.5
            if v == -2
                v = -1
            elseif v == -1
                v = 1
            else
                v = 2
            end
        else
            if v == 2
                v = 1
            elseif v == 1
                v = -1
            else
                v = -2
            end
        end
        layer[i, j] = v

    end
end

function add_remove_connection(network::Network, add::Bool)
    # With probability 0.2, set a non-zero connection to zero, or a zero connection to non-zero
    if rand() < 0.2
        indices = Tuple{Int64,Int64,Int64}[]
        sizehint!(indices, 121)
        for i in 1:8, j in 1:9
            if (network.w1[i, j] != 0) $ add
                push!(indices, (1,i,j))
                if !add
                    inc_dec(network.w1, i, j)
                end
            end
        end
        for i in 1:4, j in 1:9
            if (network.w2[i, j] != 0) $ add
                push!(indices, (2,i,j))
                if !add
                    inc_dec(network.w2, i, j)
                end
            end
        end
        for i in 1:2, j in 1:5
            if (network.w3[i, j] != 0) $ add
                push!(indices, (3,i,j))
                if !add
                    inc_dec(network.w3, i, j)
                end
            end
        end
        for i in 1:1, j in 1:3
            if (network.w4[i, j] != 0) $ add
                push!(indices, (4,i,j))
                if !add
                    inc_dec(network.w4, i, j)
                end
            end
        end

        if length(indices) > 0
            to_remove = rand(indices)
            if to_remove[1] == 1
                layer = network.w1
            elseif to_remove[1] == 2
                layer = network.w2
            elseif to_remove[1] == 3
                layer = network.w3
            else
                layer = network.w4
            end
            if add
                value = rand([-2,-1,1,2])
            else
                value = 0
            end
            layer[to_remove[2], to_remove[3]] = value
        end
    end
end

function mutated_copy(n::Network)
    network = copy(n)

    add_remove_connection(network, false)
    add_remove_connection(network, true)

    return network
end

function do_trial(trial::Int64, generations::Int64, task_switching_rate::Int64, population_size::Int64, inputs, targets, results::Array{Float64, 2})
    task = 1

    networks = Network[random_network() for i in 1:population_size] # initial population

    for generation in 1:generations
        if generation % task_switching_rate == 0
            task = 3 - task
        end

        fitnesses = Int64[network_accuracy(network, inputs, targets, task) for network in networks]
        results[trial, generation] = mean(fitnesses)
        #println("$(mean(fitnesses)), $(maximum(fitnesses))")

        new_networks = Network[]
        sizehint!(new_networks, population_size)
        for p in 1:population_size
            i1 = rand(1:population_size)
            i2 = rand(1:population_size)
            parent_index = i1
            if fitnesses[i2] > fitnesses[i1]
                parent_index = i2
            end
            push!(new_networks,  mutated_copy(networks[parent_index]))
        end
        networks = new_networks
    end

end

using PyPlot

function main()
    inputs, targets = get_targets()

    trials = 1
    generations = 50
    results = zeros(trials, generations)
    task_switching_rate = 10
    population_size = 1000

    for trial in 1:trials
        print("$trial      \r")
        do_trial(trial, generations, task_switching_rate, population_size, inputs, targets, results)
    end
    println("")

    PyPlot.plot(mean(results, 1)')
    PyPlot.show()

end

main()

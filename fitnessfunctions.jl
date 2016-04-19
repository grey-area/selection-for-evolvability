using Distributions
#using Memoize
using StatsBase
import Base.copy
import Base.==

abstract Individual
abstract FitnessFunction

type SimpleIndividual <: Individual
    f::Float64
    e::Float64
end

SimpleIndividual() = SimpleIndividual(0.0, 10.0)
Base.copy(i::SimpleIndividual) = Individual(i.f, i.e)

type SimpleFitnessFunction <: FitnessFunction
end

function mutated_copy(p::SimpleIndividual)
    q = 1.0
    f2 = p.f+ randn() * p.e
    e2 = p.e + randn() * sqrt(q)
    if e2 < 0
        e2 = 0.0001
    end
    return SimpleIndividual(f2, e2)
end

function delta_fitness_function!(fitness_function::SimpleFitnessFunction)
end

function evaluate_fitness(individual::SimpleIndividual, fitness_function::SimpleFitnessFunction)
    return individual.f
end


type ReisingerIndividual <: Individual
    bitstring::Array{Int64,1}
    mutation_mask::Array{Int64,1}
end

ReisingerIndividual() = ReisingerIndividual(rand(Distributions.Bernoulli(0.5), 100), rand(Distributions.Bernoulli(0.5), 100))
Base.copy(i::ReisingerIndividual) = ReisingerIndividual(copy(i.bitstring), copy(i,mutation_mask))

type ReisingerFitnessFunction <: FitnessFunction
    bitstring::Array{Int64,1}
    mutation_mask::Array{Int64,1}
end

ReisingerFitnessFunction() = ReisingerFitnessFunction(rand(Distributions.Bernoulli(0.5), 100), rand(Distributions.Bernoulli(0.5), 100))

function reisinger_changes(individual::Union{ReisingerIndividual, ReisingerFitnessFunction}, mutation_rate::Float64)
    mutations = rand(Distributions.Bernoulli(mutation_rate), 100)
    masked_mutations = mutations & individual.mutation_mask
    new_bitstring = individual.bitstring $ masked_mutations
    meta_mutations = rand(Distributions.Bernoulli(0.5*mutation_rate), 100)
    new_mutation_mask = individual.mutation_mask $ meta_mutations
    return new_bitstring, new_mutation_mask
end

function mutated_copy(individual::ReisingerIndividual)
    mutation_rate = 0.01
    new_bitstring, new_mutation_mask = reisinger_changes(individual, mutation_rate)
    return ReisingerIndividual(new_bitstring, new_mutation_mask)
end

function delta_fitness_function!(fitness_function::ReisingerFitnessFunction)
    mutation_rate = 0.01
    fitness_function.bitstring, fitness_function.mutation_mask = reisinger_changes(fitness_function, mutation_rate)
end

function evaluate_fitness(individual::ReisingerIndividual, fitness_function::ReisingerFitnessFunction)
    return mean(individual.bitstring .== fitness_function.bitstring)
end


type LipsonIndividual <: Individual
    w1::Array{Int64, 2}
    w2::Array{Int64, 2}
    w3::Array{Int64, 2}
    w4::Array{Int64, 2}
end

function LipsonIndividual()
    return LipsonIndividual(rand(-2:2, 8, 9), rand(-2:2, 4, 9), rand(-2:2, 2, 5), rand(-2:2, 1, 3))
end
Base.copy(network::LipsonIndividual) = LipsonIndividual(copy(network.w1), copy(network.w2), copy(network.w3), copy(network.w4))
==(n1::LipsonIndividual, n2::LipsonIndividual) = (n1.w1 == n2.w1 && n1.w2 == n2.w2 && n1.w3 == n2.w3 && n1.w4 == n2.w4)

type LipsonFitnessFunction <: FitnessFunction
    inputs::Array{Array{Int64,1}, 1}
    targets::BitArray{2}
    task::Int64
end

function LipsonFitnessFunction()
    left_objects = [1 1 1 1; 1 1 1 0; 1 0 1 0; 1 0 1 1; 1 0 0 0; 1 1 0 1; 0 0 1 0; 0 1 1 1]
    right_objects = [1 1 1 1; 1 1 1 0; 0 1 0 1; 1 0 1 1; 0 0 0 1; 1 1 0 1; 0 1 0 0; 0 1 1 1]

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

    return LipsonFitnessFunction(inputs, targets, 1)
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

function add_remove_connection(network::LipsonIndividual, add::Bool)
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

    return 0
end

function mutated_copy(n::LipsonIndividual)
    network = copy(n)

    add_remove_connection(network, false)
    add_remove_connection(network, true)

    return network
end

function delta_fitness_function!(fitness_function::LipsonFitnessFunction)
    if rand() < 0.2
        fitness_function.task = 3 - fitness_function.task
    end
end

function evaluate(network::LipsonIndividual, input::Array{Int64, 1})
    lambda = 20
    output = copy(input)

    for layer in Any[network.w1, network.w2, network.w3, network.w4]
        push!(output, 1.0)
        output = tanh(lambda * layer * output)
    end

    return output[1] > 0
end

lipson_dict = Dict{Tuple{Array{Int64, 2},Array{Int64, 2},Array{Int64, 2},Array{Int64, 2}}, Float64}()

function evaluate_fitness(network::LipsonIndividual, fitness_function::LipsonFitnessFunction)
    tup = (network.w1, network.w2, network.w3, network.w4)

    if haskey(lipson_dict, tup)
        return lipson_dict[tup]::Float64
    else
        res = convert(Float64, sum(Bool[evaluate(network, input) == fitness_function.targets[input_i, fitness_function.task] for (input_i, input) in enumerate(fitness_function.inputs)]))
        lipson_dict[tup] = res
        return res
    end
end

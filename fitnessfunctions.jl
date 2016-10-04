using Distributions
#using Memoize
using StatsBase
import Base.copy
import Base.==

abstract Individual
abstract FitnessFunction

#=
TODO
- change lipson, remove 8x8 layer
- change way mirrored mutations are calculated to match description in thesis
=#


# Simulated binary crossover with n=2
function cross_float(v1::Float64, v2::Float64, lower_limit::Float64 = -Inf, upper_limit::Float64 = Inf)
    u = rand()
    beta = u <= 0.5 ? (2u)^(1/3) : 1/(2(1-u))^(1/3)
    x = (v1 + v2) / 2
    c1 = max(min(x - beta/2 * (v2 - v1), upper_limit), lower_limit)
    c2 = max(min(x + beta/2 * (v2 - v1), upper_limit), lower_limit)
    return c1, c2
end

function cross_array(v1::Array{Int64, 1}, v2::Array{Int64, 1})
    index = rand(0:length(v1))
    c1 = [v1[1:index]; v2[index+1:end]]
    c2 = [v2[1:index]; v1[index+1:end]]
    return c1, c2
end



type SimpleIndividual <: Individual
    f::Float64
    e::Float64
end

SimpleIndividual() = SimpleIndividual(0.0, 10.0)
Base.copy(i::SimpleIndividual) = SimpleIndividual(i.f, i.e)

type SimpleFitnessFunction <: FitnessFunction
    implements_crossover::Bool
end
SimpleFitnessFunction() = SimpleFitnessFunction(true)

function mutated_copy(p::SimpleIndividual)
    q = 1.0
    f2 = p.f+ randn() * p.e
    e2 = p.e + randn() * sqrt(q)
    if e2 < 0
        e2 = 0.0001
    end
    return SimpleIndividual(f2, e2)
end

function crossover(p1::SimpleIndividual, p2::SimpleIndividual)
    c1f = p1.f; c2f = p2.f; c1e = p1.e; c2e = p2.e

    if rand() < 0.5
        c1f, c2f = cross_float(c1f, c2f)
    else
        c1e, c2e = cross_float(c1e, c2e, 0.001)
    end

    return SimpleIndividual(c1f, c1e), SimpleIndividual(c2f, c2e)
end

function delta_fitness_function!(fitness_function::SimpleFitnessFunction, problem_delta_rate::Float64, problem_delta_rate2::Float64)
end

function evaluate_fitness(individual::SimpleIndividual, fitness_function::SimpleFitnessFunction)
    return individual.f
end





type MaskIndividual <: Individual
    bitstring::Array{Int64,1}
    mutation_mask::Array{Int64,1}
end

MaskIndividual() = MaskIndividual(rand(Distributions.Bernoulli(0.5), 100), rand(Distributions.Bernoulli(0.5), 100))
Base.copy(i::MaskIndividual) = MaskIndividual(copy(i.bitstring), copy(i.mutation_mask))

type MaskFitnessFunction <: FitnessFunction
    bitstring::Array{Int64,1}
    mutation_mask::Array{Int64,1}
    implements_crossover::Bool
end

MaskFitnessFunction() = MaskFitnessFunction(rand(Distributions.Bernoulli(0.5), 100), rand(Distributions.Bernoulli(0.5), 100), true)

function mask_changes(individual::Union{MaskIndividual, MaskFitnessFunction}, mutation_rate::Float64, mutation_rate2)
    mutations = rand(Distributions.Bernoulli(mutation_rate), 100)
    masked_mutations = mutations & individual.mutation_mask
    new_bitstring = individual.bitstring $ masked_mutations
    meta_mutations = rand(Distributions.Bernoulli(mutation_rate2), 100)
    new_mutation_mask = individual.mutation_mask $ meta_mutations
    return new_bitstring, new_mutation_mask
end

function mutated_copy(individual::MaskIndividual)
    mutation_rate = 0.01
    new_bitstring, new_mutation_mask = mask_changes(individual, mutation_rate, 0.5*mutation_rate)
    return MaskIndividual(new_bitstring, new_mutation_mask)
end

function crossover(p1::MaskIndividual, p2::MaskIndividual)
    c1b = copy(p1.bitstring); c1m = copy(p1.mutation_mask); c2b = copy(p2.bitstring); c2m = copy(p2.mutation_mask)

    if rand() < 0.5
        c1b, c2b = cross_array(c1b, c2b)
    else
        c1m, c2m = cross_array(c1m, c2m)
    end

    return MaskIndividual(c1b, c1m), MaskIndividual(c2b, c2m)
end

function delta_fitness_function!(fitness_function::MaskFitnessFunction, problem_delta_rate::Float64, problem_delta_rate2::Float64)
    mutation_rate = 0.01 * problem_delta_rate
    mutation_rate2 = 0.5 * 0.01 * problem_delta_rate2
    fitness_function.bitstring, fitness_function.mutation_mask = mask_changes(fitness_function, mutation_rate, mutation_rate2)
end

function evaluate_fitness(individual::MaskIndividual, fitness_function::MaskFitnessFunction)
    return 100.0 * mean(individual.bitstring .== fitness_function.bitstring)
end






type SymmetryIndividual <: Individual
    bitstring::Array{Int64,1}
    mirrored_probability::Float64
end

SymmetryIndividual() = SymmetryIndividual(rand(Distributions.Bernoulli(0.5), 100), rand())
Base.copy(i::SymmetryIndividual) = SymmetryIndividual(copy(i.bitstring), i.mirrored_probability)

type SymmetryFitnessFunction <: FitnessFunction
    bitstring::Array{Int64,1}
    mirrored_probability::Float64
    implements_crossover::Bool
end

SymmetryFitnessFunction() = SymmetryFitnessFunction(rand(Distributions.Bernoulli(0.5), 100), rand(), true)

function symmetry_changes(individual::Union{SymmetryIndividual, SymmetryFitnessFunction}, mutation_rate::Float64, mutation_rate2::Float64)
    mutations = rand(Distributions.Bernoulli(mutation_rate), 100)
    mirrored = rand() < individual.mirrored_probability
    if mirrored
        mutations[51:end] = 0
        mutations = mutations $ reverse(mutations)
    end
    new_bitstring = individual.bitstring $ mutations

    meta_mutation = 10.0 * mutation_rate2 * (rand() - 0.5)
    new_mirrored_probability = max(min(individual.mirrored_probability + meta_mutation, 1.0), 0.0)
    return new_bitstring, new_mirrored_probability
end

function mutated_copy(individual::SymmetryIndividual)
    mutation_rate = 0.01
    new_bitstring, new_mirrored_probability = symmetry_changes(individual, mutation_rate, mutation_rate)
    return SymmetryIndividual(new_bitstring, new_mirrored_probability)
end

function crossover(p1::SymmetryIndividual, p2::SymmetryIndividual)
    c1b = copy(p1.bitstring); c1m = p1.mirrored_probability; c2b = copy(p2.bitstring); c2m = p2.mirrored_probability

    if rand() < 0.5
        c1b, c2b = cross_array(c1b, c2b)
    else
        c1m, c2m = cross_float(c1m, c2m, 0.0, 1.0)
    end

    return SymmetryIndividual(c1b, c1m), SymmetryIndividual(c2b, c2m)
end

function delta_fitness_function!(fitness_function::SymmetryFitnessFunction, problem_delta_rate::Float64, problem_delta_rate2::Float64)
    mutation_rate = 0.01 * problem_delta_rate
    mutation_rate2 = 0.01 * problem_delta_rate2
    fitness_function.bitstring, fitness_function.mirrored_probability = symmetry_changes(fitness_function, mutation_rate, mutation_rate2)
end

function evaluate_fitness(individual::SymmetryIndividual, fitness_function::SymmetryFitnessFunction)
    return 100.0 * mean(individual.bitstring .== fitness_function.bitstring)
end





type LipsonIndividual <: Individual
    w2::Array{Int64, 2}
    w3::Array{Int64, 2}
    w4::Array{Int64, 2}
end

function LipsonIndividual()
    return LipsonIndividual(rand(-2:2, 4, 9), rand(-2:2, 2, 5), rand(-2:2, 1, 3))
end
Base.copy(network::LipsonIndividual) = LipsonIndividual(copy(network.w2), copy(network.w3), copy(network.w4))
==(n1::LipsonIndividual, n2::LipsonIndividual) = (n1.w2 == n2.w2 && n1.w3 == n2.w3 && n1.w4 == n2.w4)

type LipsonFitnessFunction <: FitnessFunction
    inputs::Array{Array{Int64,1}, 1}
    targets::BitArray{2}
    task::Int64
    implements_crossover::Bool
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

    return LipsonFitnessFunction(inputs, targets, 1, false)
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
            if to_remove[1] == 2
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

function delta_fitness_function!(fitness_function::LipsonFitnessFunction, problem_delta_rate::Float64, problem_delta_rate2::Float64)
    threshold = 1/(1+exp(-problem_delta_rate)) - 0.5
    if rand() < threshold || problem_delta_rate2 < problem_delta_rate
        fitness_function.task = 3 - fitness_function.task
    end
end

function evaluate(network::LipsonIndividual, input::Array{Int64, 1})
    lambda = 20
    output = copy(input)

    for layer in Any[network.w2, network.w3, network.w4]
        push!(output, 1.0)
        output = tanh(lambda * layer * output)
    end

    return output[1] > 0
end

lipson_dict = Dict{Tuple{Array{Int64, 2},Array{Int64, 2},Array{Int64, 2}}, Float64}()

function evaluate_fitness(network::LipsonIndividual, fitness_function::LipsonFitnessFunction)
    tup = (network.w2, network.w3, network.w4)

    if haskey(lipson_dict, tup)
        return lipson_dict[tup]::Float64
    else
        res = 100.0 * convert(Float64, mean(Bool[evaluate(network, input) == fitness_function.targets[input_i, fitness_function.task] for (input_i, input) in enumerate(fitness_function.inputs)]))
        lipson_dict[tup] = res
        return res
    end
end

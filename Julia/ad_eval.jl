using Random
using BenchmarkTools
using Zygote
using ForwardDiff
using ReverseDiff
using Enzyme

# Define the benchmark function structure
struct BenchmarkFunction
    n::Int               # input dimension               
    m::Int               # output dimension             
    num_operations::Int  # number of sin/cos steps
    r::Vector{Vector{Int}}  
    s::Vector{Vector{Int}}  
end

function BenchmarkFunction(n::Int, m::Int, num_ops::Int; seed::Int=0)
    rng = MersenneTwister(seed)
    r_data = Vector{Vector{Int}}(undef, m)
    s_data = Vector{Vector{Int}}(undef, m)
    for i in 1:m
        r_data[i] = [rand(rng, 0:n-1) for _ in 1:(num_ops + 1)]
        s_data[i] = [rand(rng, 0:1) for _ in 1:num_ops]
    end
    return BenchmarkFunction(n, m, num_ops, r_data, s_data)
end

# The raw function evaluation produces an m-dimensional output.
function call_raw(bf::BenchmarkFunction, x::AbstractVector)
    return [ begin
        rr = bf.r[i]
        ss = bf.s[i]
        tmp = x[rr[1] + 1]
        for j in 1:bf.num_operations
            if ss[j] == 0
                tmp = sin(cos(tmp) + x[rr[j+1] + 1])
            elseif ss[j] == 1
                tmp = cos(sin(tmp) + x[rr[j+1] + 1])
            else
                error("Operation not supported.")
            end
        end
        tmp
    end for i in 1:bf.m ]
end

function derivative_enzyme(bf::BenchmarkFunction, x::Vector{Float64})
    return Enzyme.jacobian(Enzyme.Reverse, x -> call_raw(bf, x), x)
end

function derivative_enzyme_forward(bf::BenchmarkFunction, x::Vector{Float64})
   return Enzyme.jacobian(Enzyme.Forward, x -> call_raw(bf, x), x)
end

function derivative_zygote(bf::BenchmarkFunction, x::AbstractVector)
    return Zygote.jacobian(x -> call_raw(bf, x), x)
end

function derivative_forwarddiff(bf::BenchmarkFunction, x::AbstractVector)
    return ForwardDiff.jacobian(x -> call_raw(bf, x), x)
end

function derivative_reversediff(bf::BenchmarkFunction, x::AbstractVector)
    return ReverseDiff.jacobian(x -> call_raw(bf, x), x)
end

# Pack all evaluation conditions in one structure.
struct EvaluationConditionPack
    bf::BenchmarkFunction
    enzyme_reverse::Function
    enzyme_forward::Function
    zygote::Function
    forwardDiffJL::Function
    reverseDiffJL::Function
end

function EvaluationConditionPack(n::Int, m::Int, o::Int; seed::Int=0)
    bf = BenchmarkFunction(n, m, o; seed=seed)
    enzyme_reverse   = x -> derivative_enzyme(bf, x)
    enzyme_forward   = x -> derivative_enzyme_forward(bf, x)
    zygote   = x -> derivative_zygote(bf, x)
    forwardDiffJL    = x -> derivative_forwarddiff(bf, x)
    reverseDiffJL    = x -> derivative_reversediff(bf, x)
    return EvaluationConditionPack(bf, enzyme_reverse, enzyme_forward,
                                   zygote, 
                                   forwardDiffJL, reverseDiffJL)
end

# Benchmark each method.
function run_experiment(n::Int, m::Int, o::Int)
    println("\nRunning Experiment with (n, m, o) = ($n, $m, $o)")
    ec = EvaluationConditionPack(n, m, o; seed=1234)
    enzyme_rev_engine = ec.enzyme_reverse
    enzyme_fd_engine = ec.enzyme_forward
    zygote_engine = ec.zygote
    rev_engine = ec.reverseDiffJL
    fd_engine = ec.forwardDiffJL
    x = rand(n)
    println("\nBenchmarking Enzyme Reverse")
    J_enzyme_rev = @btime $enzyme_rev_engine($x) seconds=2
    println("Enzyme Reverse: ", J_enzyme_rev)
    
    println("\nBenchmarking Enzyme Forward")
    J_enzyme_fwd = @btime $enzyme_fd_engine($x) seconds=2
    println("Enzyme Forward execution time: ", J_enzyme_fwd)
    
    println("\nBenchmarking Zygote")
    J_zygote_rev = @btime $zygote_engine($x) seconds=2
    println("Zygote Reverse Jacobian:       ", J_zygote_rev)

   
    
    println("\nBenchmarking ReverseDiffJL")
    J_forwarddiff = @btime $rev_engine($x) seconds=2
    #println("ReverseDiffJL Jacobian:  ", J_forwarddiff)
    
    println("\nBenchmarking ForwardDiffJL")
    J_reversediff = @btime $fd_engine($x) seconds=2
    println("ForwardDiffJL Jacobian:  ", J_reversediff)
end

function main()
    run_experiment(100, 1, 1000)
    run_experiment(1, 100, 1000)
    run_experiment(10, 10, 1000)
    #run_experiment(50, 50, 1000)
end

main()


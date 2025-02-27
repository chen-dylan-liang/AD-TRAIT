#!/bin/bash

echo "===== Starting Benchmark Suite ====="
date

# Run Rust benchmarks
echo "\n===== Running Rust Benchmarks ====="
cd Rust
bash eval1.bash
cd ..

# Run Julia benchmarks
echo "\n===== Running Julia Benchmarks ====="
cd Julia
julia ad_eval.jl
cd ..

# Run C++ benchmarks
echo "\n===== Running C++ Benchmarks ====="
cd C++/build
./AD
cd ../..

echo "\n===== All Benchmarks Completed ====="
date
echo "Results can be found in their respective directories"

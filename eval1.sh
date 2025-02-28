#!/bin/bash

echo "===== Starting Benchmark Suite ====="
date

# Run Rust benchmarks
echo "\n===== Running Rust Benchmarks ====="
cd Rust
./eval1.sh
cd ..

# Run C++ benchmarks
echo "\n===== Running C++ Benchmarks ====="
cd C++
./eval1.sh
cd ..


# Run Julia benchmarks
echo "\n===== Running Julia Benchmarks ====="
cd Julia
./eval1.sh
cd ..

# Run Python benchmarks
echo "\n===== Running Python Benchmarks ====="
cd Python
./eval1.sh
cd ..

echo "\n===== All Benchmarks Completed ====="
date
echo "Results can be found in their respective directories"

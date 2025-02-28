#include "function.hpp"
#include <autodiff/forward/real/eigen.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace autodiff;
using namespace Eigen;

std::pair<VectorXd, MatrixXd> compute_forward_ad(const BenchmarkFunction2& func, VectorXreal& x);
std::pair<VectorXd, MatrixXd> compute_reverse_ad(const BenchmarkFunction2& func, VectorXvar& x);

struct BenchmarkResult {
    int n;
    int m;
    int o;
    double forward_time;
    double reverse_time;
};

void run_benchmark(int n, int m, int o, int num_trials, std::vector<BenchmarkResult>& results) {
    double total_forward_time = 0.0;
    double total_reverse_time = 0.0;

    for (int trial = 0; trial < num_trials; ++trial) {
        BenchmarkFunction2 func(n, m, o);
        VectorXreal fwd_x = VectorXreal::Random(n);
        VectorXvar rev_x(n);

        for (int i = 0; i < n; ++i)
            rev_x[i] = fwd_x[i].val();

        // Forward AD timing
        auto start = std::chrono::high_resolution_clock::now();
        auto [F_forward, J_forward] = compute_forward_ad(func, fwd_x);
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>
            (std::chrono::high_resolution_clock::now() - start);
        total_forward_time += duration.count();

        // Reverse AD timing
        start = std::chrono::high_resolution_clock::now();
        auto [F_reverse, J_reverse] = compute_reverse_ad(func, rev_x);
        duration = std::chrono::duration_cast<std::chrono::microseconds>
            (std::chrono::high_resolution_clock::now() - start);
        total_reverse_time += duration.count();
    }

    // Calculate averages
    double avg_forward_time = total_forward_time / num_trials;
    double avg_reverse_time = total_reverse_time / num_trials;

    // Store results
    BenchmarkResult result = {n, m, o, avg_forward_time, avg_reverse_time};
    results.push_back(result);

    // Print to console
    std::cout << "Parameters: n=" << n << ", m=" << m << ", o=" << o << std::endl;
    std::cout << "Average Forward AD Time: " << avg_forward_time << "μs\n";
    std::cout << "Average Reverse AD Time: " << avg_reverse_time << "μs\n";
    std::cout << "-----------------------------------------\n";
}

void save_to_csv(const std::vector<BenchmarkResult>& results, const std::string& filename) {
    std::ofstream file(filename);

    // Write header
    file << "n,m,o,forward_time_us,reverse_time_us\n";

    // Write data
    for (const auto& result : results) {
        file << result.n << ","
             << result.m << ","
             << result.o << ","
             << result.forward_time << ","
             << result.reverse_time << "\n";
    }

    file.close();
    std::cout << "Results saved to " << filename << std::endl;
}

int main() {
    // Number of trials per parameter combination
    const int NUM_TRIALS = 100;
    std::vector<BenchmarkResult> results;

    // First set: varying input dimension with fixed output dimension
    std::vector<std::pair<int, int>> params1;
    for (int n : {1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
                  550, 600, 650, 700, 750, 800, 850, 900, 950, 1000}) {
        params1.push_back({n, 1});
    }

// Run benchmarks for first parameter set
    for (const auto& [n, m] : params1) {
        run_benchmark(n, m, 1000, NUM_TRIALS, results);
    }

// Second set: combinations of input and output dimensions
    std::vector<std::pair<int, int>> params2;
    for (int n : {1, 10, 20, 30, 40, 50}) {
            params2.push_back({n, n});
    }

// Run benchmarks for second parameter set
    for (const auto& [n, m] : params2) {
        run_benchmark(n, m, 1000, NUM_TRIALS, results);
    }

    // Save results to CSV file
    save_to_csv(results, "../autodiff_benchmark_results.csv");

    return 0;
}
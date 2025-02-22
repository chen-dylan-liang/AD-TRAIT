#include "function.hpp"
#include <autodiff/forward/real/eigen.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include <chrono>
#include <iostream>

using namespace autodiff;
using namespace Eigen;

std::pair<VectorXd, MatrixXd> compute_forward_ad(const BenchmarkFunction2& func, VectorXreal& x);
std::pair<VectorXd, MatrixXd> compute_reverse_ad(const BenchmarkFunction2& func, VectorXvar& x);

int main() {
    int n,m,o;
    std::cout<<"Please enter n, m and o"<<std::endl;
    std::cin>>n>>m>>o;
    int num_pass = 1000;
    double tfd=0, trev=0;
    for (int i = 0; i < num_pass; i++) {
        BenchmarkFunction2 func(n, m, o);
        VectorXreal fwd_x = VectorXreal::Random(n);
        VectorXvar rev_x(n);
        for (int i = 0; i < n; ++i)
            rev_x[i] = fwd_x[i].val();
        auto start = std::chrono::high_resolution_clock::now();
        auto [F_forward, J_forward] = compute_forward_ad(func, fwd_x);
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>
            (std::chrono::high_resolution_clock::now() - start);
        tfd += duration.count();
        start = std::chrono::high_resolution_clock::now();
        auto [F_reverse, J_reverse] = compute_reverse_ad(func, rev_x);
        duration = std::chrono::duration_cast<std::chrono::microseconds>
            (std::chrono::high_resolution_clock::now() - start);
       trev += duration.count();
    }
    std::cout << "Forward AD Time: " << tfd/num_pass << "μs\n";
    //<< "F = \n" << F_forward << "\nJ = \n" << J_forward << "\n\n";
    std::cout << "Reverse AD Time: " << trev/num_pass << "μs\n";
    //<< "F = \n" << F_reverse << "\nJ = \n" << J_reverse << "\n";
    return 0;
}
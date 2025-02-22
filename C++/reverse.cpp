#include "function.hpp"
#include <autodiff/reverse/var/eigen.hpp>
#include <iostream>

using namespace autodiff;
using namespace Eigen;

std::pair<VectorXd, MatrixXd> compute_reverse_ad(const BenchmarkFunction2& func, VectorXvar& x) {
    int n = func.n, m = func.m;


    MatrixXd J(m, n);
    VectorXd F(m);

    for(int i = 0; i < m; ++i) {
        var y = func.scalar_output(x, i);
        F(i) = val(y);
        VectorXd g = gradient(y, x);
        J.row(i) = g;
    }
    return {F, J};
}
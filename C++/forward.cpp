#include "function.hpp"
#include <autodiff/forward/real/eigen.hpp>

using namespace autodiff;
using namespace Eigen;

std::pair<VectorXd, MatrixXd> compute_forward_ad(const BenchmarkFunction2& func, VectorXreal& x) {
    int n = func.n, m = func.m;

    VectorXreal F;
    double y[m * n];
    Eigen::Map<MatrixXd> J(y, m, n);

    auto f = [&](const VectorXreal& x) -> VectorXreal {
        VectorXreal result(m);
        for(int i = 0; i < m; ++i)
            result(i) = func.scalar_output(x, i);
        return result;
    };

    jacobian(f, autodiff::wrt(x), autodiff::at(x), F, J);
    return {F.cast<double>(), J};
}

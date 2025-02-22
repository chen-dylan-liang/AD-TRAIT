#pragma once
#include <vector>
#include <random>
#include <Eigen/Dense>

class BenchmarkFunction2 {
public:
    int n, m, num_operations;
    std::vector<std::vector<int>> r, s;

    BenchmarkFunction2(int n, int m, int num_operations);

    template<typename T>
    T scalar_output(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x, int output_idx) const {
        const auto &rr = r[output_idx];
        const auto &ss = s[output_idx];
        T tmp = x(rr[0]);
        for (int j = 0; j < num_operations; ++j) {
            if (ss[j] == 0)
                tmp = sin(cos(tmp) + x(rr[j+1]));
            else if (ss[j] == 1)
                tmp = cos(sin(tmp) + x(rr[j+1]));
        }
        return tmp;
    }
};

inline BenchmarkFunction2::BenchmarkFunction2(int n, int m, int num_operations)
    : n(n), m(m), num_operations(num_operations), r(m), s(m) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist_r(0, n - 1);
    std::uniform_int_distribution<int> dist_s(0, 1);

    for (int i = 0; i < m; ++i) {
        r[i].resize(num_operations + 1);
        s[i].resize(num_operations);
        for (int j = 0; j < num_operations + 1; ++j)
            r[i][j] = dist_r(rng);
        for (int j = 0; j < num_operations; ++j)
            s[i][j] = dist_s(rng);
    }
}
use apollo_rust_linalg_adtrait::V;
use ad_trait::differentiable_function::{ReverseAD, ForwardAD, ForwardADMulti};

use nalgebra::DVector;
use rand::Rng;

pub struct BenchmarkFunction {
    n: usize,
    m: usize,
    num_operations: usize,
    r: Vec<Vec<usize>>,
    s: Vec<Vec<i32>>,
}

impl BenchmarkFunction {
    pub fn new(n: usize, m: usize, num_operations: usize) -> Self {
        let mut rng = rand::rng();
        let mut r = Vec::with_capacity(m);
        let mut s = Vec::with_capacity(m);

        for _ in 0..m {
            // Create a vector of length (num_operations + 1) with random indices in the range [0, n)
            let r_vec: Vec<usize> = (0..(num_operations + 1))
                .map(|_| rng.random_range(0..n))
                .collect();
            // Create a vector of length num_operations with random values 0 or 1.
            let s_vec: Vec<i32> = (0..num_operations)
                .map(|_| rng.random_range(0..2))
                .collect();
            r.push(r_vec);
            s.push(s_vec);
        }
        BenchmarkFunction {
            n,
            m,
            num_operations,
            r,
            s,
        }
    }

    /// Performs the computation on the input tensor `x` represented as a DVector.
    ///
    /// For each output function (total `m`), it starts with an element of `x` (chosen using
    /// the first index in the corresponding `r` vector) and then performs a sequence of operations.
    /// The operations are chosen based on the values in the corresponding `s` vector:
    /// - If s[j] == 0: compute sin(cos(tmp) + x[rr[j + 1]])
    /// - If s[j] == 1: compute cos(sin(tmp) + x[rr[j + 1]])
    pub fn call_raw(&self, x: &DVector<f64>) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.m);

        for i in 0..self.m {
            let rr = &self.r[i];
            let ss = &self.s[i];
            // Start with the tensor value at the first random index.
            let mut tmp = x[rr[0]];
            // Apply each operation in sequence.
            for j in 0..self.num_operations {
                let val = x[rr[j + 1]];
                tmp = match ss[j] {
                    0 => (tmp.cos() + val).sin(),
                    1 => (tmp.sin() + val).cos(),
                    _ => panic!("Operation not supported"),
                };
            }
            out.push(tmp);
        }

        out
    }

    /// Returns the input dimension.
    pub fn input_dim(&self) -> usize {
        self.n
    }

    /// Returns the output dimension.
    pub fn output_dim(&self) -> usize {
        self.m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_function() {
        let n = 10;
        let m = 3;
        let num_operations = 2;
        // Create a BenchmarkFunction instance.
        let bf = BenchmarkFunction::new(n, m, num_operations);
        // Create a sample input tensor as a DVector of f64 values.
        let x = DVector::from_vec((0..n).map(|i| i as f64).collect());
        let output = bf.call_raw(&x);
        // Verify that the number of outputs matches `m`.
        assert_eq!(output.len(), bf.output_dim());
    }
}


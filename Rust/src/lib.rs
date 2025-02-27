mod eval1;
mod eval2;
mod csv_utils;

use std::char::MAX;
use std::cmp::{max, min};
use std::time::Instant;
use ad_trait::differentiable_block::DifferentiableBlock;
use ad_trait::differentiable_function::{ForwardAD, ForwardADMulti, ReverseAD};
use ad_trait::forward_ad::adfn::adfn;
use apollo_rust_linalg_adtrait::{ApolloDVectorTrait, V};
use eval1::{BenchmarkFunctionNalgebra, DCBenchmarkFunctionNalgebra};
use csv_utils::{ write_data, calculate_stats};
pub struct EvaluationConditionPack<const N: usize> {
    //pub finite_differencing: DifferentiableBlock<DifferentiableFunctionClassBenchmarkFunction2, FiniteDifferencing>,
    pub f: BenchmarkFunctionNalgebra,
    pub forward_ad: DifferentiableBlock<DCBenchmarkFunctionNalgebra, ForwardAD>,
    pub reverse_ad: DifferentiableBlock<DCBenchmarkFunctionNalgebra, ReverseAD>,
    pub mc_forward_ad: DifferentiableBlock<DCBenchmarkFunctionNalgebra, ForwardADMulti<adfn<N>>>,
}
impl<const N:usize> EvaluationConditionPack<N> {
    pub fn new(n: usize, m: usize, o: usize) -> Self {
        let f = BenchmarkFunctionNalgebra::new(n, m, o);
        Self {
            f: f.clone(),
            forward_ad: DifferentiableBlock::new_with_tag(DCBenchmarkFunctionNalgebra, ForwardAD::new(), f.clone(), f.clone()),
            reverse_ad: DifferentiableBlock::new_with_tag(DCBenchmarkFunctionNalgebra, ReverseAD::new(), f.clone(), f.clone()),
            mc_forward_ad: DifferentiableBlock::new_with_tag(DCBenchmarkFunctionNalgebra, ForwardADMulti::<adfn<N>>::new(), f.clone(), f.clone()),
        }
    }
}

pub fn benchmark_eval1<const N:usize>(pack:&EvaluationConditionPack<N>, passes:usize) {
    println!("Evaluation 1 running for n={}, m={}", pack.f.n, pack.f.m);
    let mut runtime= vec![Vec::<f64>::new();3];
    let approaches = ["FAD-ad trait-Rust", "RAD-ad trait-Rust", "FAD-SIMD-ad trait-Rust"];
    for p in 0..passes {
        println!("Pass {p} running...");
        let x_nalgebra = V::<f64>::new_random_with_range(pack.f.n, -1.0, 1.0);
        let x_vec: &[f64] = x_nalgebra.as_slice();
        for i in 0..approaches.len() {
            let approach = approaches[i];
            let start = Instant::now();
            let (f_val, jacobian) = match approach {
                "FAD-ad trait-Rust" => pack.forward_ad.derivative(x_vec),
                "RAD-ad trait-Rust" => pack.reverse_ad.derivative(x_vec),
                "FAD-SIMD-ad trait-Rust" => pack.mc_forward_ad.derivative(x_vec),
                        _ => panic!("Unknown approach: {}", approach), };
            let duration = start.elapsed().as_secs_f64()*1000.0*1000.0;
            runtime[i].push(duration);
}
}
    for i in 0..runtime.len() {
        let (avg, std) =calculate_stats(&runtime[i]);
        write_data("results/eval1.csv", approaches[i], pack.f.n, pack.f.m, avg, std);
    }
}



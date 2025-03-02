pub mod eval1;
pub mod eval2;
pub mod csv_utils;
pub mod burn;

use std::char::MAX;
use std::cmp::{max, min, Reverse};
use std::time::Instant;
use ad_trait::AD;
use ad_trait::differentiable_block::DifferentiableBlock;
use ad_trait::differentiable_function::{DifferentiableFunctionTrait, FiniteDifferencing, ForwardAD, ForwardADMulti, ReverseAD};
use ad_trait::forward_ad::adfn::adfn;
use ad_trait::reverse_ad::adr::GlobalComputationGraph;
use apollo_rust_linalg_adtrait::{ApolloDVectorTrait, V};
use eval1::{BenchmarkFunctionNalgebra, DCBenchmarkFunctionNalgebra};
use csv_utils::{ write_data, calculate_stats};
use crate::csv_utils::calculate_stats_without_outliers;
use crate::eval2::{simple_pseudoinverse_newtons_method_ik, BenchmarkIK, DCBenchmarkIK};

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

pub fn benchmark_eval2(){
    let passes = 500;
    let ik = BenchmarkIK::<f64>::new();
    let mut durs_fad =Vec::<f64>::new();
    let mut durs_rad =Vec::<f64>::new();
    let mut durs_fd =Vec::<f64>::new();
    let mut durs_mcfad =Vec::<f64>::new();
    for i in 0..passes {
        println!("Pass {i} running...");
        let init_cond = V::<f64>::new_random_with_range(ik.num_inputs(),-0.2,0.2);
        durs_fad.push(simple_pseudoinverse_newtons_method_ik(ForwardAD::new(), init_cond.clone(), 10000,0.01, 0.01));
        durs_rad.push(simple_pseudoinverse_newtons_method_ik(ReverseAD::new(), init_cond.clone(), 10000,0.01, 0.01));
        durs_fd.push(simple_pseudoinverse_newtons_method_ik(FiniteDifferencing::new(), init_cond.clone(), 10000,0.01, 0.01));
        durs_mcfad.push(simple_pseudoinverse_newtons_method_ik(ForwardADMulti::<adfn<24>>::new(), init_cond.clone(), 10000,0.01, 0.01));
    }
    println!("Forward AD:, (avg_time, std_time)={:?}", calculate_stats(&durs_fad));
    println!("Reverse AD: (avg_time, std_time)={:?}", calculate_stats(&durs_rad));
    println!("Finite Diff:, (avg_time, std_time)={:?}", calculate_stats(&durs_fd));
    println!("Multi Channel Forward AD:,  (avg_time, std_time)={:?}", calculate_stats(&durs_mcfad));
}



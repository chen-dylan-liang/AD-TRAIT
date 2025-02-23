use ad_trait_eval::EvaluationConditionPack;
use std::io;
use std::time::Instant;
use apollo_rust_linalg_adtrait::{ApolloDVectorTrait, V};
fn benchmark(n: usize, m:usize, o:usize, passes:usize) -> [f64; 6] {
    let mut durations = [0.0; 6];
    let approaches = ["forward_ad", "reverse_ad", "mc8", "mc16","mc32", "mc1000"];
    for p in 0..passes {
        println!("Pass {p} running...");
        let pack = EvaluationConditionPack::new(n, m, o);
        let x_nalgebra = V::<f64>::new_random_with_range(n, -1.0, 1.0);
        let x_vec: &[f64] = x_nalgebra.as_slice();
        for i in 0..approaches.len() {
            let approach = approaches[i];
            let start = Instant::now();
            let (_, d_gt) = match approach {
                "forward_ad" => pack.forward_ad.derivative(x_vec),
                "reverse_ad" => pack.reverse_ad.derivative(x_vec),
                "mc8" => pack.forward_ad_multi_8.derivative(x_vec),
                "mc16" => pack.forward_ad_multi_16.derivative(x_vec),
                "mc32" => pack.forward_ad_multi_32.derivative(x_vec),
                "mc1000" => pack.forward_ad_multi_1000.derivative(x_vec),
                _ => panic!("Unknown approach: {}", approach),
            };
            let duration = start.elapsed().as_secs_f64();
            durations[i]+=duration;
        }
    }
    for i in 0..durations.len() {
        durations[i]=durations[i]/passes as f64;
        println!("{}:{}",approaches[i],durations[i]);
    }
    durations
}
fn main() {
    println!("Please enter n, m, o, and passes");
    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Failed to read");
    let nums: Vec<usize> = input
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let durations=benchmark(nums[0], nums[1], nums[2], nums[3]);
}
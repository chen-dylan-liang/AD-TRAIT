use ad_trait_eval::EvaluationConditionPack;
use std::io;
use std::time::Instant;
use apollo_rust_linalg_adtrait::{ApolloDVectorTrait, V};
use ad_trait_eval::INPUT_DIM;

fn benchmark(n: usize, m:usize, o:usize, passes:usize) -> [f64; 3] {
    let mut durations = [0.0; 3];
    let approaches = ["forward_ad", "reverse_ad", "mc_forward_ad",];
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
                "mc_forward_ad" => pack.mc_forward_ad.derivative(x_vec),
                _ => panic!("Unknown approach: {}", approach),
            };
            let duration = start.elapsed().as_secs_f64()*1000.0*1000.0;
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
    println!("Please enter m, o, and passes");
    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Failed to read");
    let nums: Vec<usize> = input
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let durations=benchmark(INPUT_DIM, nums[0], nums[1], nums[2]);
}
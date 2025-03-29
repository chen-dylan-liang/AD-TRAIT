use ad_trait::differentiable_block::DifferentiableBlock;
use apollo_rust_linalg_adtrait::{ApolloDVectorTrait, V};
use ad_trait_eval::{benchmark_eval1, benchmark_eval2, EvaluationConditionPack};


const INPUT_DIM: usize = 100;
const OUTPUT_DIM: usize = 1;
const MAX_THREADS: usize = 128;
const N_CHANNELS: usize = if INPUT_DIM < MAX_THREADS{INPUT_DIM} else {MAX_THREADS};


fn run_eval1() {
    let pack = EvaluationConditionPack::<N_CHANNELS>::new(INPUT_DIM, OUTPUT_DIM, 1000);
    benchmark_eval1::<N_CHANNELS>(&pack, 100);
}

fn run_eval2() {
    benchmark_eval2();
}

fn main(){
    run_eval1();
}
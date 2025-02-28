use apollo_rust_linalg_adtrait::{ApolloDVectorTrait, V};
use ad_trait_eval::{benchmark_eval1, EvaluationConditionPack};


const INPUT_DIM: usize = 50;
const OUTPUT_DIM: usize = 50;
const MAX_THREADS: usize = 128;
const N_CHANNELS: usize = if INPUT_DIM < MAX_THREADS{INPUT_DIM} else {MAX_THREADS};


fn main() {
    let pack = EvaluationConditionPack::<N_CHANNELS>::new(INPUT_DIM, OUTPUT_DIM, 1000);
    benchmark_eval1::<N_CHANNELS>(&pack, 100);
}
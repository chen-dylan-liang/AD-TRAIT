use ad_trait::differentiable_function::{DifferentiableFunctionTrait, ForwardAD, ReverseAD, ForwardADMulti};
use ad_trait::function_engine::FunctionEngine;
use apollo_rust_linalg_adtrait::{ApolloDVectorTrait, V};
use ad_trait::forward_ad::adfn::adfn;
//use ad_trait_eval::{benchmark_eval1, benchmark_eval2, EvaluationConditionPack, live_demo::ForwardKinematics};
use ad_trait_eval::live_demo::ForwardKinematics;

const INPUT_DIM: usize = 50;
const OUTPUT_DIM: usize = 1;
const MAX_THREADS: usize = 128;
const N_CHANNELS: usize = if INPUT_DIM < MAX_THREADS{INPUT_DIM} else {MAX_THREADS};

/*
fn run_eval1() {
    let pack = EvaluationConditionPack::<N_CHANNELS>::new(INPUT_DIM, OUTPUT_DIM, 1000);
    benchmark_eval1::<N_CHANNELS>(&pack, 100);
}

fn run_eval2() {
    benchmark_eval2();
}*/

pub fn get_jacobian(){
    // choose ad_method among ForwardAD, ReverseAD, ForwardADMulti::<adfn<N>>, etc according to your needs
    let ad_method = ForwardADMulti::<adfn<16>>::new();
    // construct an ad engine providing derivatives
    let ad_engine = FunctionEngine::new(
        ForwardKinematics::new(), // used for function value evaluations
        ForwardKinematics::new(),  // used for function derivative accumulations
        ad_method);
    // randomly sample an input
    let input = V::<f64>::new_random_with_range(ForwardKinematics::<f64>::new().num_inputs(),-0.2,0.2);
    // get jacobian
    println!("Jacobian at {} is {:?}", input, ad_engine.derivative(input.as_slice()));
}

fn main(){
    get_jacobian()
}
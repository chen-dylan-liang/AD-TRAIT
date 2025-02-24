mod eval1;
mod eval2;

use std::char::MAX;
use std::cmp::{max, min};
use ad_trait::differentiable_block::DifferentiableBlock;
use ad_trait::differentiable_function::{ForwardAD, ForwardADMulti, ReverseAD};
use ad_trait::forward_ad::adfn::adfn;
use eval1::{BenchmarkFunctionNalgebra, DCBenchmarkFunctionNalgebra};
pub const INPUT_DIM: usize = 1000;
const MAX_THREADS: usize = 128;
const N_CHANNELS: usize = if INPUT_DIM < MAX_THREADS{INPUT_DIM} else {MAX_THREADS};
pub struct EvaluationConditionPack {
    //pub finite_differencing: DifferentiableBlock<DifferentiableFunctionClassBenchmarkFunction2, FiniteDifferencing>,
    pub forward_ad: DifferentiableBlock<DCBenchmarkFunctionNalgebra, ForwardAD>,
    pub reverse_ad: DifferentiableBlock<DCBenchmarkFunctionNalgebra, ReverseAD>,
    pub mc_forward_ad: DifferentiableBlock<DCBenchmarkFunctionNalgebra, ForwardADMulti<adfn<N_CHANNELS>>>,
}
impl EvaluationConditionPack {
    pub fn new(n: usize, m: usize, o: usize) -> Self {
        let f = BenchmarkFunctionNalgebra::new(n, m, o);
        Self {
            //finite_differencing: DifferentiableBlock::new_with_tag(DifferentiableFunctionClassBenchmarkFunction2, FiniteDifferencing::new(), f.clone(), f.clone()),
            forward_ad: DifferentiableBlock::new_with_tag(DCBenchmarkFunctionNalgebra, ForwardAD::new(), f.clone(), f.clone()),
            reverse_ad: DifferentiableBlock::new_with_tag(DCBenchmarkFunctionNalgebra, ReverseAD::new(), f.clone(), f.clone()),
            mc_forward_ad: DifferentiableBlock::new_with_tag(DCBenchmarkFunctionNalgebra, ForwardADMulti::<adfn<N_CHANNELS>>::new(), f.clone(), f.clone()),
        }
    }
}





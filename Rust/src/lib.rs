mod eval1;
mod eval2;

use ad_trait::differentiable_block::DifferentiableBlock;
use ad_trait::differentiable_function::{ForwardAD, ForwardADMulti, ReverseAD};
use ad_trait::forward_ad::adfn::adfn;
use eval1::{BenchmarkFunctionNalgebra, DCBenchmarkFunctionNalgebra};

pub struct EvaluationConditionPack {
    //pub finite_differencing: DifferentiableBlock<DifferentiableFunctionClassBenchmarkFunction2, FiniteDifferencing>,
    pub forward_ad: DifferentiableBlock<DCBenchmarkFunctionNalgebra, ForwardAD>,
    pub reverse_ad: DifferentiableBlock<DCBenchmarkFunctionNalgebra, ReverseAD>,
    pub forward_ad_multi_8: DifferentiableBlock<DCBenchmarkFunctionNalgebra, ForwardADMulti<adfn<8>>>,
    pub forward_ad_multi_16: DifferentiableBlock<DCBenchmarkFunctionNalgebra, ForwardADMulti<adfn<16>>>,
    pub forward_ad_multi_32: DifferentiableBlock<DCBenchmarkFunctionNalgebra, ForwardADMulti<adfn<32>>>,
    pub forward_ad_multi_1000: DifferentiableBlock<DCBenchmarkFunctionNalgebra, ForwardADMulti<adfn<1000>>>,
}
impl EvaluationConditionPack {
    pub fn new(n: usize, m: usize, o: usize) -> Self {
        let f = BenchmarkFunctionNalgebra::new(n, m, o);
        Self {
            //finite_differencing: DifferentiableBlock::new_with_tag(DifferentiableFunctionClassBenchmarkFunction2, FiniteDifferencing::new(), f.clone(), f.clone()),
            forward_ad: DifferentiableBlock::new_with_tag(DCBenchmarkFunctionNalgebra, ForwardAD::new(), f.clone(), f.clone()),
            reverse_ad: DifferentiableBlock::new_with_tag(DCBenchmarkFunctionNalgebra, ReverseAD::new(), f.clone(), f.clone()),
            forward_ad_multi_8: DifferentiableBlock::new_with_tag(DCBenchmarkFunctionNalgebra, ForwardADMulti::<adfn<8>>::new(), f.clone(), f.clone()),
            forward_ad_multi_16: DifferentiableBlock::new_with_tag(DCBenchmarkFunctionNalgebra, ForwardADMulti::<adfn<16>>::new(), f.clone(), f.clone()),
            forward_ad_multi_32: DifferentiableBlock::new_with_tag(DCBenchmarkFunctionNalgebra, ForwardADMulti::<adfn<32>>::new(), f.clone(), f.clone()),
            forward_ad_multi_1000: DifferentiableBlock::new_with_tag(DCBenchmarkFunctionNalgebra, ForwardADMulti::<adfn<1000>>::new(), f.clone(), f.clone()),
        }
    }
}





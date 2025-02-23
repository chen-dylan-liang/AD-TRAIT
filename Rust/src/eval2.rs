use ad_trait::AD;
use ad_trait::differentiable_function::{DifferentiableFunctionClass, DifferentiableFunctionTrait};
use apollo_rust_linalg_adtrait::V;
use apollo_rust_spatial_adtrait::vectors::{V3, V6};
use apollo_rust_spatial_adtrait::lie::se3_implicit_quaternion::LieGroupISE3q;
use apollo_rust_spatial_adtrait::quaternions::UQ;
use apollo_rust_robotics_core_adtrait::ChainNalgebraADTrait;
use apollo_rust_modules::ResourcesRootDirectory;
use apollo_rust_robotics_adtrait::ToChainNalgebraADTrait;
use nalgebra::UnitQuaternion;
use crate::eval1::BenchmarkFunctionNalgebra;

pub struct BenchmarkIK<T:AD>{
    chain: ChainNalgebraADTrait<T>,
    target_foot1_pos: V3<T>,
    target_foot2_pos: V3<T>,
    target_foot3_pos: V3<T>,
    target_foot4_pos: V3<T>,
    target_ee_pos: LieGroupISE3q<T>,
}

impl <T:AD> BenchmarkIK<T>{
    pub fn new()->Self{
        let dir=ResourcesRootDirectory::new_from_default_apollo_robots_dir();
        let chain:ChainNalgebraADTrait<T> = dir.get_subdirectory("b1z1").to_chain_nalgebra_adtrait();
        Self{
            chain,
            target_foot1_pos: V3::new(0.45.into(), (-0.2).into(), 0.0.into()),
            target_foot2_pos: V3::new(0.45.into()  , 0.2.into()  , 0.0.into()  ),
            target_foot3_pos: V3::new((-0.2).into()   , (-0.2).into()  , 0.0.into()  ),
            target_foot4_pos: V3::new((-0.2).into()  , 0.2.into()  , 0.0.into()  ),
            target_ee_pos: LieGroupISE3q::from_exponential_coordinates(&V6::new(0.0.into()  , 0.0.into()  , 0.0.into()  ,
                                                                                     0.3.into()  , 0.0.into()  , 1.15.into()  ))
        }
    }
}

impl <T:AD> DifferentiableFunctionTrait<T> for BenchmarkIK<T>{
    fn call(&self, inputs: &[T], freeze: bool) -> Vec<T> {
        let res = self.chain.fk(&V::<T>::from_column_slice(inputs));
        let t1 = (res[10].0.translation.vector - self.target_foot1_pos).norm();
        let t2 = (res[17].0.translation.vector - self.target_foot2_pos).norm();
        let t3 = (res[24].0.translation.vector - self.target_foot3_pos).norm();
        let t4 = (res[31].0.translation.vector - self.target_foot4_pos).norm();
        let t5 = LieGroupISE3q::new(res[39].0.inverse()*self.target_ee_pos.0).ln().vee().norm();
        Vec::from([t1*t1, t2*t2, t3*t3, t4*t4, t5*t5])
    }

    fn num_inputs(&self) -> usize {
        self.chain.num_dofs()
    }

    fn num_outputs(&self) -> usize {
        5
    }
}

pub struct DCBenchmarkIK;
impl DifferentiableFunctionClass for DCBenchmarkIK {
    type FunctionType<T: AD> = BenchmarkIK<T>;
}

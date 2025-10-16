use ad_trait::AD;
use ad_trait::differentiable_function::{DifferentiableFunctionTrait};
use apollo_rust_linalg_adtrait::{ApolloDMatrixTrait, ApolloDVectorTrait, V};
use apollo_rust_robotics_core_adtrait::ChainNalgebraADTrait;
use apollo_rust_modules::ResourcesRootDirectory;
use apollo_rust_robotics_adtrait::ToChainNalgebraADTrait;
use apollo_rust_spatial_adtrait::vectors::ApolloVector3ADTrait;

#[derive(Clone)]
pub struct ForwardKinematics<T:AD>{
    // ChainNalgebra is the representation for robots in apollo-rust
    chain: ChainNalgebraADTrait<T>,
}

impl <T:AD> ForwardKinematics<T>{
    pub fn new()->Self{
        // load the robot (b1 robot dog with z1 arm mounted) from the specified directory
        let dir=ResourcesRootDirectory::new_from_default_apollo_robots_dir();
        let chain:ChainNalgebraADTrait<T> = dir.get_subdirectory("b1z1").
            to_chain_nalgebra_adtrait::<f64>().to_other_ad_type::<T>();
        Self{
            chain,
        }
    }
}

impl <T:AD> DifferentiableFunctionTrait<T> for ForwardKinematics<T>{
    const NAME: &'static str = "ForwardKinematics";
    // do forward kinematics in this call routine
    // get the robot dog's base's translation
    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        Vec::<T>::from(self.chain.fk(&V::from_column_slice(inputs))[1].0.translation.vector.as_slice())
    }

    // specify number of inputs
    fn num_inputs(&self) -> usize {
        self.chain.num_dofs()
    }

    // specify number of outputs
    fn num_outputs(&self) -> usize { 3 }
}

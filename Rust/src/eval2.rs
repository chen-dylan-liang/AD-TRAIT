use std::ops::Mul;
use std::time::Instant;
use ad_trait::AD;
use ad_trait::ADNumMode::ForwardAD;
use ad_trait::function_engine::FunctionEngine;
use ad_trait::differentiable_function::{DerivativeMethodTrait, DifferentiableFunctionTrait};
use ad_trait::reverse_ad::adr::adr;
use apollo_rust_linalg_adtrait::{ApolloDMatrixTrait, V};
use apollo_rust_spatial_adtrait::vectors::{V3, V6};
use apollo_rust_spatial_adtrait::lie::se3_implicit_quaternion::LieGroupISE3q;
use apollo_rust_spatial_adtrait::quaternions::UQ;
use apollo_rust_robotics_core_adtrait::ChainNalgebraADTrait;
use apollo_rust_modules::ResourcesRootDirectory;
use apollo_rust_robotics_adtrait::ToChainNalgebraADTrait;
use nalgebra::{Isometry3, UnitQuaternion};
use crate::eval1::BenchmarkFunctionNalgebra;
use apollo_rust_linalg_adtrait::{M, SVDType, SVDResult};
use apollo_rust_lie_adtrait::{LieGroupElement, LieAlgebraElement};
use apollo_rust_spatial_adtrait::lie::h1::ApolloUnitQuaternionH1LieTrait;
use apollo_rust_spatial_adtrait::isometry3::{ApolloIsometry3Trait, I3};
#[derive(Clone)]
pub struct BenchmarkIK<T:AD>{
    chain: ChainNalgebraADTrait<T>,
}

impl <T:AD> BenchmarkIK<T>{
    pub fn new()->Self{
        let dir=ResourcesRootDirectory::new_from_default_apollo_robots_dir();
        let chain:ChainNalgebraADTrait<T> = dir.get_subdirectory("b1z1").to_chain_nalgebra_adtrait::<f64>().to_other_ad_type::<T>();
        Self{
            chain,
        }
    }
    pub fn to_other_ad_type<T2: AD>(&self) -> BenchmarkIK<T2> {
        BenchmarkIK { chain: self.chain.to_other_ad_type::<T2>()}
    }
}

impl <T:AD> DifferentiableFunctionTrait<T> for BenchmarkIK<T>{
    const NAME: &'static str = "InverseKinematics";
    fn call(&self, inputs: &[T], freeze: bool) -> Vec<T> {
        let res = self.chain.fk(&V::from_column_slice(inputs));
        let t1 = (res[10].0.translation.vector - V3::new(0.45.into(), (-0.2).into(), 0.0.into())).norm();
        let t2 = (res[17].0.translation.vector - V3::new(0.45.into()  , 0.2.into()  , 0.0.into()  )).norm();
        let t3 = (res[24].0.translation.vector - V3::new((-0.2).into()   , (-0.2).into()  , 0.0.into()  )).norm();
        let t4 = (res[31].0.translation.vector - V3::new((-0.2).into()  , 0.2.into()  , 0.0.into()  )).norm();
        let target_pose = LieGroupISE3q::new(Isometry3::from_slices_euler_angles(&[0.3.into(), 0.0.into(), 1.15.into()], &[0.0.into(), 0.0.into(), 0.0.into()]));
        let t5 = (res[39].0.translation.vector - target_pose.0.translation.vector).norm();
        let t6 = res[39].0.rotation.to_lie_group_h1().displacement(&target_pose.0.rotation.to_lie_group_h1()).ln().vee().norm();

        vec![t1*t1, t2*t2, t3*t3, t4*t4, t5*t5, t6*t6]
            }

    fn num_inputs(&self) -> usize {
        self.chain.num_dofs()
    }

    fn num_outputs(&self) -> usize {
        6
    }
}

/*
def simple_pseudoinverse_newtons_method_for_experiment_2(condition, init_condition=None,
                                                         record_num_f_calls=False, change_step_length=False) -> Experiment2Result:
    """
    :param record_num_f_calls:
    :param init_condition:
    :param condition: should be from EvaluationConditionPack3
    :return:
    :change_step_length: only set to True when using SPSA
    """
    tl.set_backend(condition.backend.to_string())
    iterates = []
    num_f_calls = []
    if init_condition is None:
        q = tl.tensor(np.random.uniform(-0.2, 0.2, (24,)))
    else:
        q = tl.tensor(init_condition)
    iterates.append(tl.to_numpy(q).tolist())
    start = time.time()
    y = condition.call(q)

    for i in range(10000):
        delta_q = T2.pinv(condition.derivative(q)) @ y
        if record_num_f_calls:
            num_f_calls.append(condition.d.num_f_calls)
        if not change_step_length:
            q = q - 0.01 * delta_q
        else:
            q = q - 1.0/(i+1.0)*delta_q
        y = condition.call(q)
        print(i, y)
        iterates.append(tl.to_numpy(q).tolist())
        if tl.norm(y) < 0.01:
            break

    end = time.time() - start
    return Experiment2Result(end, iterates, num_f_calls)
 */
fn pseudo_inverse<T: AD>(mat: &M<T>, eps: T) -> M<T> {
    let svd = mat.singular_value_decomposition(SVDType::Full);
    let s = svd.singular_values();
    let u = svd.u();
    let vt = svd.vt();
    let mut s_pinv = M::<T>::zeros(vt.nrows(), u.ncols());
    let min_dim = std::cmp::min(s_pinv.nrows(), s_pinv.ncols());
    for i in 0..std::cmp::min(s.len(), min_dim) {
        if s[i].abs() > eps {
            s_pinv[(i, i)] = T::one().div(s[i]);
        } else {
            s_pinv[(i, i)] = T::zero();
        }
    }

    vt.transpose() * s_pinv * u.transpose()
}

pub fn simple_pseudoinverse_newtons_method_ik<E:DerivativeMethodTrait>(ad_method: E, init_cond: V<f64>, max_iter: usize, step_length: f64, threshold: f64) -> f64{
    let ik = BenchmarkIK::<f64>::new();
    let ik_d = ik.to_other_ad_type::<E::T>();
    let ad_engine = FunctionEngine::new(ik, ik_d, ad_method);
    let mut q = init_cond;
    let start = Instant::now();
    //let mut y = V::<f64>::from(ad_engine.call(q.as_slice()));
    for i in 0..max_iter {
        let (_y, jacobian) = ad_engine.derivative(q.as_slice());
        let y = V::<f64>::from(_y);
        if y.norm() < threshold {break;}
        let delta_q = pseudo_inverse(&jacobian, 1e-12)*(&y);
        q = q - step_length*delta_q;
    }
    let duration = start.elapsed().as_secs_f64();
    duration
}

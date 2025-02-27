use std::time::Instant;
use ad_trait::AD;
use ad_trait::differentiable_block::DifferentiableBlock;
use ad_trait::differentiable_function::{DerivativeMethodTrait, DifferentiableFunctionClass, DifferentiableFunctionTrait};
use apollo_rust_linalg_adtrait::{ApolloDMatrixTrait, V};
use apollo_rust_spatial_adtrait::vectors::{V3, V6};
use apollo_rust_spatial_adtrait::lie::se3_implicit_quaternion::LieGroupISE3q;
use apollo_rust_spatial_adtrait::quaternions::UQ;
use apollo_rust_robotics_core_adtrait::ChainNalgebraADTrait;
use apollo_rust_modules::ResourcesRootDirectory;
use apollo_rust_robotics_adtrait::ToChainNalgebraADTrait;
use nalgebra::UnitQuaternion;
use crate::eval1::BenchmarkFunctionNalgebra;
use apollo_rust_linalg_adtrait::{M, SVDType, SVDResult};
use apollo_rust_lie_adtrait::{LieGroupElement, LieAlgebraElement};

#[derive(Clone)]
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
fn pseudo_inverse<T:AD>(mat: &M<T>, eps: T) -> M<T> {
    let svd = mat.singular_value_decomposition(SVDType::Full);
    let s = svd.singular_values();
    let u = svd.u();
    let vt = svd.vt();

    let mut s_pinv= M::<T>::zeros(vt.nrows(), u.nrows());

    for i in 0..s.len() {
        if s[i] > eps {
           s_pinv[(i, i)] = T::one().div(s[i]);
        } else {
            s_pinv[(i, i)] = T::zero();
        }
    }
    vt.transpose() * s_pinv * u.transpose()
}

pub fn simple_pseudoinverse_newtons_method_ik<E:DerivativeMethodTrait<T=f64>>(ad_method: E, init_cond: V<f64>, max_iter: usize, step_length: f64, threshold: f64) -> f64{
    let ik = BenchmarkIK::<f64>::new();
    let ad_engine = DifferentiableBlock::new_with_tag(DCBenchmarkIK, ad_method, ik.clone(), ik.clone());
    let mut q = init_cond;
    let start = Instant::now();
    let mut y = V::<f64>::from(ad_engine.call(q.as_slice()));
    for i in 0..max_iter {
        let (_, jacobian) = ad_engine.derivative(q.as_slice());
        let delta_q = pseudo_inverse(&jacobian, 1e-10)*y;
        q = q - step_length*delta_q;
        y = V::<f64>::from(ad_engine.call(q.as_slice()));
        println!("{}: {:?}",i,y);
        if y.norm() < threshold {break;}
    }
    let duration = start.elapsed().as_secs_f64();
    duration
}
use apollo_rust_linalg_adtrait::{V, M, ApolloDVectorTrait};
use rand::Rng;
use ad_trait::AD;
use ad_trait::differentiable_function::{DifferentiableFunctionClass, DifferentiableFunctionTrait};

#[derive(Clone, Debug)]
pub struct BenchmarkFunctionVec {
    n: usize,
    m: usize,
    o: usize,
    r: Vec<Vec<usize>>,
    s: Vec<Vec<usize>>
}

impl BenchmarkFunctionVec {
    pub fn new(n: usize, m: usize, o: usize) -> Self {
        let mut r = vec![];
        let mut s = vec![];

        let mut rng = rand::rng();
        for _ in 0..m {
            let rr: Vec<usize> = (0..=o).map(|_| rng.random_range(0..n)).collect();
            let ss: Vec<usize> = (0..o).map(|_| rng.random_range(0..=1)).collect();
            r.push(rr);
            s.push(ss);
        }

        Self {
            n,
            m,
            o,
            r,
            s,
        }
    }
}

impl<T: AD> DifferentiableFunctionTrait<T> for BenchmarkFunctionVec {
    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        let mut out = vec![];

        for i in 0..self.m {
            let rr = &self.r[i];
            let ss = &self.s[i];

            let mut tmp = inputs[rr[0]];
            for j in 0..self.o {
                if ss[j] == 0 {
                    tmp = (tmp.cos() * inputs[rr[j+1]]).sin();
                } else if ss[j] == 1 {
                    tmp = (tmp.sin() * inputs[rr[j+1]]).cos();
                } else {
                    panic!("not supported")
                }
            }
            out.push(tmp);
        }

        return out;
    }

    fn num_inputs(&self) -> usize {
        self.n
    }

    fn num_outputs(&self) -> usize {
        self.m
    }
}

pub struct DCBenchmarkFunctionVec;
impl DifferentiableFunctionClass for DCBenchmarkFunctionVec {
    type FunctionType<T: AD> = BenchmarkFunctionVec;
}

#[derive(Clone, Debug)]
pub struct BenchmarkFunctionNalgebra{
    n: usize,
    m: usize,
    o: usize,
    r: V<usize>,
    s: V<usize>,
}

impl BenchmarkFunctionNalgebra {
    pub fn new(n: usize, m: usize, o: usize) -> Self {
        let mut r = V::<usize>::zeros(m*(o+1));
        let mut s = V::<usize>::zeros(m*o);
        let mut rng = rand::rng();
        for i in 0..m {
            for j in 0..=o {r[i*(o+1)+j]=rng.random_range(0..n)}
            for j in 0..o {s[i*o+j]=rng.random_range(0..=1)}
        }
        Self{n,m,o,r,s}
    }
}

impl<T: AD> DifferentiableFunctionTrait<T> for BenchmarkFunctionNalgebra {
    fn call(&self, inputs: &[T], freeze: bool) -> Vec<T> {
        let mut out = V::<T>::zeros(self.m);

        for i in 0..self.m {
            out[i] = inputs[self.r[i*self.o]];
            for j in 0..self.o {
                if self.s[i * self.o + j] == 0 {
                    out[i] = (out[i].cos() * inputs[self.r[i * self.o + j + 1]]).sin();
                } else if self.s[i * self.o + j] == 1 {
                    out[i] = (out[i].sin() * inputs[self.r[i * self.o + j + 1]]).cos();
                } else {
                    panic!("not supported")
                }
            }
        }
        //out
        out.data.as_vec().to_vec()
    }

    fn num_inputs(&self) -> usize {
        self.n
    }

    fn num_outputs(&self) -> usize {
        self.m
    }
}

pub struct DCBenchmarkFunctionNalgebra;
impl DifferentiableFunctionClass for DCBenchmarkFunctionNalgebra {
    type FunctionType<T: AD> = BenchmarkFunctionNalgebra;
}

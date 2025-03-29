use burn::backend::{Autodiff, Wgpu, NdArray};
//use burn::backend::
use burn::prelude::Tensor;
use burn::tensor::Distribution;
use rand::Rng;

type Backend = Autodiff<NdArray>;
pub fn benchmark(x: &Tensor<Backend, 1>, o: usize) -> Tensor<Backend, 1>
{
    let n = x.dims()[0];
    let mut rng = rand::thread_rng();
    let r: Vec<usize> = (0..(o + 1)).map(|_| rng.gen_range(0..n)).collect();
    let s: Vec<usize> = (0..o).map(|_| rng.gen_range(1..=2)).collect();

    let mut tmp = x.clone().slice([r[0]..r[0]+1]);

    for j in 0..o {
        let x_rj = x.clone().slice([r[j+1]..r[j+1]+1]);
        tmp = if s[j] == 1 {
            (tmp.cos() + x_rj).sin()
        } else {
            (tmp.sin() + x_rj).cos()
        };
    }

    tmp
}

pub fn get_random_walk(
    n: usize,
    num_waypoints: usize,
    step_length: f64,
) -> Vec<Tensor<Backend, 1>> {
    let device = Default::default();
    let mut curr = Tensor::random([n], Distribution::Default, &device).require_grad();
    let mut out = Vec::with_capacity(num_waypoints);
    out.push(curr.clone());

    for _ in 1..num_waypoints {
        curr = Tensor::random([n], Distribution::Default, &device).require_grad();
        out.push(curr.clone());
    }

    out
}
use std::time::Instant;
use ad_trait_eval::burn::{benchmark, get_random_walk};

fn main() {
    let n = 1000;
    let m = 1;
    let o = 1000;

    // let x: Tensor<Backend, 1> = Tensor::random([n], Distribution::Default, &Default::default()).require_grad();
    let xx = get_random_walk(n, 10, 0.1);
    let start = Instant::now();
    for _ in 0..m {
        for x in &xx {
            let result = benchmark(x, o);
            let grads = result.backward();
            let x_grad = x.grad(&grads).unwrap();
            // println!("gradient = {}", x_grad);
        }
    }
    println!("{:?}", start.elapsed());
}
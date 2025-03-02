use std::time::Instant;
use ad_trait_eval::burn::{benchmark, get_random_walk};
use ad_trait_eval::csv_utils::write_data;

fn main() {
    // First part: Run with n=(1,50,100,150,...,1000) and m=1
    println!("Running first part: n values (1,50,100,150,...,1000), m=1");
    let n_values = [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000];

    for &n in &n_values {
        let m = 1;
        let o = 1000;

        println!("Testing with n={}, m={}", n, m);
        let xx = get_random_walk(n, 100, 0.1);
        let start = Instant::now();
        for _ in 0..m {
            for x in &xx {
                let result = benchmark(x, o);
                let grads = result.backward();
                let x_grad = x.grad(&grads).unwrap();
                // println!("gradient = {}", x_grad);
            }
        }
        let duration = start.elapsed();
        write_data("results/eval1_burn.csv","burn",n,m,duration.as_secs_f64()/100.0,0.0);
    }

    // Second part: Run with n,m combinations of (1,10,20,30,40,50)
    println!("\nRunning second part: n,m combinations of (1,10,20,30,40,50)");
    let parameter_values = [1, 10, 20, 30, 40, 50];
    for &n in &parameter_values {
            let o = 1000;
            let m=n;
            println!("Testing with n={}, m={}", n, m);
            let xx = get_random_walk(n, 100, 0.1);
            let start = Instant::now();
            for _ in 0..m {
                for x in &xx {
                    let result = benchmark(x, o);
                    let grads = result.backward();
                    let x_grad = x.grad(&grads).unwrap();
                    // println!("gradient = {}", x_grad);
                }
            }
        let duration = start.elapsed();
        write_data("results/eval1_burn.csv","burn",n,m,duration.as_secs_f64()/100.0,0.0);

    }
}

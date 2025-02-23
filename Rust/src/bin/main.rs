use ad_trait_eval::EvaluationConditionPack;


fn benchmark(n: usize, m:usize, o:usize){
    let pack = EvaluationConditionPack::new(n,m,o);
}
fn main() {
    benchmark(10, 10, 1000);
}
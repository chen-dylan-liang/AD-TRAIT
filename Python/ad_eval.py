import time

import jax
import numpy as np
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.derivative_method_tensorly import \
    DerivativeMethodForwardADJax, DerivativeMethodReverseADJax, DerivativeMethodReverseADPytorch
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_engine_tensorly import \
    FunctionEngine
from apollo_toolbox_py.apollo_py.apollo_py_differentiation.apollo_py_differentiation_tensorly.function_tensorly \
import BenchmarkFunction2
from jax import Device
from apollo_toolbox_py.apollo_py.extra_tensorly_backend import Backend, Device, DType
import tensorly as tl

class EvaluationConditionPack1:
    """
    For Experiment 1
    """

    def __init__(self, n: int, m: int, o: int):
        f = BenchmarkFunction2(n, m, o)
        self.forward_ad_jax_jit_gpu = FunctionEngine(f, DerivativeMethodForwardADJax(), backend=Backend.JAX,
                                                              device=Device.CUDA, jit_compile_d=True)
        self.reverse_ad_jax_jit_gpu = FunctionEngine(f, DerivativeMethodReverseADJax(), backend=Backend.JAX,
                                                              device=Device.CUDA, jit_compile_d=True)
        self.reverse_ad_pytorch = FunctionEngine(f, DerivativeMethodReverseADPytorch(), backend=Backend.PyTorch)



import csv
import os


def run_experiment(n, m, num_passes=5):
    print(f"\nRunning Experiment with (n, m) = ({n}, {m})")
    methods = ["forward_ad_jax_jit_gpu", "reverse_ad_jax_jit_gpu"]
    #methods = ["reverse_ad_pytorch"]
    results = {}

    pack = EvaluationConditionPack1(n, m, 1000)

    for method in methods:
        print(f"Benchmarking {method}")
        engine = getattr(pack, method)
        tl.set_backend(engine.backend.to_string())
        x = np.random.uniform(-1, 1, (n,))
        x_tensor = tl.tensor(x)
        print("Warmup")
        # Warm-up run
        engine.d_call(x_tensor)
        print("Finish Warmup")
        # Timed runs
        runtime = 0
        for i in range(num_passes):
            x = np.random.uniform(-1, 1, (n,))
            x_tensor = tl.tensor(x)
            start = time.time()
            engine.d_call(x_tensor)
            runtime += time.time() - start

        avg_runtime = runtime / num_passes
        results[method] = avg_runtime
        print(f"{method} average runtime: {avg_runtime:.6f} seconds")

    # Write results to CSV
    csv_file = "autodiff_benchmark_results_gpu1.csv"
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["approach", "n", "m", "time"])

        for method, runtime in results.items():
            writer.writerow([method, n, m, runtime])

    return results


if __name__ == "__main__":
    # First set: varying input dimension with fixed output dimension
    for n in [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
              550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]:
        run_experiment(n, 1)

    # Second set: combinations of input and output dimensions
    for n in [1, 10, 20, 30, 40, 50]:
        run_experiment(n, n)








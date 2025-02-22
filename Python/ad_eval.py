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

        self.forward_ad_jax = FunctionEngine(f, DerivativeMethodForwardADJax(), backend=Backend.JAX)
        self.reverse_ad_jax = FunctionEngine(f, DerivativeMethodReverseADJax(), backend=Backend.JAX)
        self.forward_ad_jax_jit_compiled = FunctionEngine(f, DerivativeMethodForwardADJax(), backend=Backend.JAX, device=Device.CPU,
                                                          jit_compile_d=True)
        self.reverse_ad_jax_jit_compiled = FunctionEngine(f, DerivativeMethodReverseADJax(), backend=Backend.JAX,device=Device.CPU,
                                                          jit_compile_d=True)
        self.forward_ad_jax_jit_compiled_gpu = FunctionEngine(f, DerivativeMethodForwardADJax(), backend=Backend.JAX,
                                                              device=Device.CUDA, jit_compile_d=True)
        self.reverse_ad_jax_jit_compiled_gpu = FunctionEngine(f, DerivativeMethodReverseADJax(), backend=Backend.JAX,
                                                              device=Device.CUDA, jit_compile_d=True)
        self.reverse_ad_pytorch = FunctionEngine(f, DerivativeMethodReverseADPytorch(), backend=Backend.PyTorch)
        self.forward_ad_jax_gpu = FunctionEngine(f, DerivativeMethodForwardADJax(), backend=Backend.JAX,
                                                 device=Device.CUDA)
        self.reverse_ad_jax_gpu = FunctionEngine(f, DerivativeMethodReverseADJax(), backend=Backend.JAX,
                                                 device=Device.CUDA)
        self.reverse_ad_pytorch_gpu = FunctionEngine(f, DerivativeMethodReverseADPytorch(), backend=Backend.PyTorch,
                                                     device=Device.CUDA)

if __name__ == "__main__":
    n=100
    m=1
    num_passes=100
    fd_jax_jit = 0
    rev_jax_jit = 0
    import os

    os.environ["JAX_PLATFORM_NAME"] = "cpu"

    import jax
    print(jax.devices())  # Should list a CPU device
    for i in range(num_passes):
        print("Start Pass {}".format(i))
        pack = EvaluationConditionPack1(n, m, 1000)
        tl.set_backend(pack.forward_ad_jax_jit_compiled.backend.to_string())
        x= tl.tensor(np.random.uniform(-1, 1, (n,)))
        start = time.time()
        pack.forward_ad_jax_jit_compiled.d_call(x)
        fd_jax_jit += time.time() - start
        start = time.time()
        pack.reverse_ad_jax_jit_compiled.d_call(x)
        rev_jax_jit += time.time() - start

    print(rev_jax_jit / num_passes)
    print(fd_jax_jit / num_passes)





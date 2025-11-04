
from setuptools import setup, Extension
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(
    name='KTransformersOps',
    ext_modules=[
        CUDAExtension(
            'KTransformersOps', [
                'custom_gguf/dequant.cu',
                'binding.cpp',
                'gptq_marlin/gptq_marlin.cu',
                # 'gptq_marlin_repack.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-Xcompiler', '-fPIC',
                ]
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)

from setuptools import setup, Extension
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# setup marlin gemm
setup(name='KTransformersOps',
      ext_modules=[
          CUDAExtension('KTransformersOps', [
                'custom_gguf/dequant.cu',
                'binding.cpp',
                'gptq_marlin/gptq_marlin.cu',
               # 'gptq_marlin_repack.cu',
              ],#extra_compile_args={'cxx': ['-g'], 'nvcc': ['-ccbin D:\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\Llvm\\x64\\bin\\clang++.exe']}
                        )
      ],
      cmdclass={'build_ext': BuildExtension
})


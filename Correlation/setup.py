from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name="DispCorr",
      ext_modules=[cpp_extension.CppExtension("DispCorr", ['DispCorr.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
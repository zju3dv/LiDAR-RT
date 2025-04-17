import os
import sysconfig
import glob, shutil
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


# Determine the OptiX SDK path
OPTIX_HOME = os.environ.get('OPTIX_HOME')
if OPTIX_HOME is None or not os.path.exists(OPTIX_HOME):
    OPTIX_HOME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/optix")
    if not os.path.exists(OPTIX_HOME):
        raise ValueError("Please set the OPTIX_HOME environment variable to the path to the OptiX SDK")
OPTIX_HOME = os.path.join(OPTIX_HOME, "include")
print(f"Using OptiX SDK at {OPTIX_HOME}")

# Determine the GLM path
GLM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm")
if not os.path.exists(GLM_PATH):
    raise ValueError("Please set the GLM_PATH environment variable to the path to the GLM library")
print(f"Using GLM at {GLM_PATH}")


# Custom build extension to build the OptiX tracing kernel
class CustomBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def build_extensions(self):
        # Record the python package source directory
        pkg_source = os.path.dirname(os.path.abspath(__file__))

        # Run the original build_extensions
        super().build_extensions()

        # Use CMake to build the OptiX tracing kernel ptx files
        os.makedirs(pkg_source + '/build', exist_ok=True)
        os.system(f'cd {pkg_source}/build && cmake .. && cmake --build .')
        pkg_target = sysconfig.get_path('purelib') + '/diff_lidar_tracer'

        # Create the target directory if it does not exist
        if not os.path.exists(pkg_target):
            os.makedirs(pkg_target, exist_ok=True)

        # Copy the `.ptx` files to the python package
        ptx_files = glob.glob(os.path.join(pkg_source, 'build', 'ptx', '*.ptx'))
        os.makedirs(pkg_target, exist_ok=True)
        for ptx_file in ptx_files:
            shutil.copy(ptx_file, pkg_target)



# Setup for the python package
setup(
    name="diff_lidar_tracer",
    packages=['diff_lidar_tracer'],
    version='0.0.1',
    ext_modules=[
        CUDAExtension(
            name="diff_lidar_tracer._C",
            sources=[
                "optix_tracer/common.cpp",
                "optix_tracer/optix_wrapper.cpp",
                "trace_surfels.cpp",
                "ext.cpp"
            ],
            # extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]},
            include_dirs=[OPTIX_HOME, GLM_PATH]
        ),
    ],
    cmdclass={
        'build_ext': CustomBuildExtension
    }
)

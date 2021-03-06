cmake_minimum_required(VERSION 3.14)
project(pynvcloth)

set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)
include("SetOutputPaths")

# On Windows, dlls must all be copied to the same final output directory
# We must do this explicitly, since we have subprojects (from 
# add_subdirectory), which use a different build directory
SetDllOutputPath("${CMAKE_CURRENT_BINARY_DIR}/bin")
SetLibOutputPath("${CMAKE_CURRENT_BINARY_DIR}/lib")
SetExeOutputPath("${CMAKE_CURRENT_BINARY_DIR}/bin")

# Add NvCloth as a subproject. This requires us to specify some build
# variables.
set(GW_DEPS_ROOT $ENV{GW_DEPS_ROOT})
set(PYNV_COMPILE_DEFS )
if(WIN32)
  set(PX_SELECT_COMPONENTS PxFoundation)
  find_package(PxShared REQUIRED)
  find_package(Cuda 10 REQUIRED)
  find_package(NvCloth REQUIRED)
  set(NV_CLOTH_ENABLE_CUDA 1)
  set(NV_CLOTH_ENABLE_DX11 1)
  list(APPEND PYNV_COMPILE_DEFS
	PYNV_CLOTH_ENABLE_DX11=1
	PYNV_CLOTH_ENABLE_CUDA=1
  )
  add_subdirectory(${NVCLOTH_ROOT_DIR}/compiler/cmake/windows "${CMAKE_CURRENT_BINARY_DIR}/NvCloth_bin")
else()
  # TODO: add cuda support here, too
  find_package(NvCloth REQUIRED)
  set(NV_CLOTH_ENABLE_CUDA 0)
  set(NV_CLOTH_ENABLE_DX11 0)
  list(APPEND PYNV_COMPILE_DEFS
	PYNV_CLOTH_ENABLE_DX11=0
	PYNV_CLOTH_ENABLE_CUDA=0
  )
  add_subdirectory(${NVCLOTH_ROOT_DIR}/compiler/cmake/linux "${CMAKE_CURRENT_BINARY_DIR}/NvCloth_bin")
endif()

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

# Specify the build targets
add_executable(hello_nvcloth
        src/hello_nvcloth.cpp
        src/CallbackImplementations.cpp
        )
target_include_directories(hello_nvcloth
        # TODO(daniel): consider wrapping these transitive deps in an interface library
        PRIVATE ${PXSHARED_ROOT_DIR}/include/foundation
        PRIVATE ${PXSHARED_ROOT_DIR}/include
        PRIVATE ${PXSHARED_ROOT_DIR}/src/foundation/include
        PRIVATE ${CUDA_INCLUDE_DIRS}
        PRIVATE ${NVCLOTH_ROOT_DIR}/include
        PRIVATE ${NVCLOTH_ROOT_DIR}/extensions/include)
target_link_libraries(hello_nvcloth
       PUBLIC d3dcompiler.lib d3d11.lib dxgi.lib comctl32.lib ${CUDA_LIBRARIES} NvCloth)
target_compile_definitions(hello_nvcloth
  PRIVATE ${PYNV_COMPILE_DEFS})

add_subdirectory(external/pybind11)
pybind11_add_module(example
        src/example.cpp
        )
        
pybind11_add_module(pynvcloth
        src/nvcloth_wrapper.cpp
        src/CallbackImplementations.cpp
        )
target_include_directories(pynvcloth
        # TODO(daniel): consider wrapping these transitive deps in an interface library
        PRIVATE ${PXSHARED_ROOT_DIR}/include/foundation
        PRIVATE ${PXSHARED_ROOT_DIR}/include
        PRIVATE ${PXSHARED_ROOT_DIR}/src/foundation/include
        PRIVATE ${CUDA_INCLUDE_DIRS}
        PRIVATE ${NVCLOTH_ROOT_DIR}/include
        PRIVATE ${NVCLOTH_ROOT_DIR}/extensions/include)
target_link_libraries(pynvcloth
       PUBLIC d3dcompiler.lib d3d11.lib dxgi.lib comctl32.lib ${CUDA_LIBRARIES} NvCloth Eigen3::Eigen)
target_compile_definitions(pynvcloth 
  PRIVATE ${PYNV_COMPILE_DEFS})

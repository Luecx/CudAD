cmake_minimum_required(VERSION 3.21)
project(CudAD VERSION 1.0 LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)
set(CMAKE_CUDA_STANDARD 14)

file(GLOB_RECURSE SRCS src/*.cu src/*.h)

add_executable(CudAD ${SRCS})

#add_executable(CudAD src/main.cu
#        src/config/config.cu
#        src/operations/add/add_host.cu
#        src/operations/add/add_kernel.cu)

set_target_properties(CudAD PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(CudAD ${CUDA_LIBRARIES})
target_link_libraries(CudAD ${CUDA_CUBLAS_LIBRARIES})

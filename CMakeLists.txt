cmake_minimum_required(VERSION 3.17)
project(CudAD VERSION 1.0 LANGUAGES CUDA CXX)

find_package(CUDA)
find_package(Threads)
set(CMAKE_CUDA_STANDARD 17)

file(GLOB_RECURSE SRCS src/*.cu src/*.cpp src/*.h)


add_executable(CudAD ${SRCS} src/main.cu)

#add_executable(CudAD src/main.cu
#        src/config/config.cu
#        src/operations/add/add_host.cu
#        src/operations/add/add_kernel.cu)

set_property(TARGET CudAD PROPERTY CUDA_ARCHITECTURES OFF)
set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")

set_target_properties(CudAD PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -openmp ")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 ")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -dlto")


# target_compile_options(CudAD PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--std=c++17>)

target_link_libraries(CudAD ${CUDA_LIBRARIES})
target_link_libraries(CudAD ${CUDA_CUBLAS_LIBRARIES})
target_link_libraries(CudAD ${CMAKE_THREAD_LIBS_INIT})

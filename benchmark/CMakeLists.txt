set(CTEST_BUILD_NAME ${ROOT_ARCHITECTURE}-${CMAKE_BUILD_TYPE})
enable_testing()

CB_ADD_GBENCHMARK(Simple Simple.cpp)
CB_ADD_GBENCHMARK(AlgorithmicComplexity AlgorithmicComplexity.cpp)
CB_ADD_GBENCHMARK(ArrayExpressionTemplates ArrayExpressionTemplates.cpp)
if (CLAD_ENABLE_ENZYME_BACKEND)
  CB_ADD_GBENCHMARK(EnzymeCladComparison EnzymeCladComparison.cpp)
endif(CLAD_ENABLE_ENZYME_BACKEND)

include(FetchContent)
FetchContent_Declare(
  tapenade_kit
  GIT_REPOSITORY https://gitlab.inria.fr/tapenade/tapenade.git
  
  # We use a commit hash from the 'develop' branch because
  # the 'disk offloading' feature is not yet available in the stable release of tapenade.
  GIT_TAG        6f8cb04a4168cac5f6242cdeb65f6816c413f4ee 
  GIT_PROGRESS   TRUE      
)
FetchContent_Declare(
    LULESH-CUDA
    GIT_REPOSITORY https://github.com/wsmoses/Enzyme-GPU-Tests.git
    GIT_TAG        9be62d07ad09ee22d03e1b8f8be28c6852eb7f70
  )
FetchContent_MakeAvailable(
  tapenade_kit
  LULESH-CUDA
)

# Build the support library for the stack from the fetched source
add_library(tapenade_support STATIC ${tapenade_kit_SOURCE_DIR}/ADFirstAidKit/adStack.c)
target_include_directories(tapenade_support PUBLIC ${tapenade_kit_SOURCE_DIR}/ADFirstAidKit)

target_compile_definitions(tapenade_support PRIVATE 
      ADSTACK_MAX_SPACES=1        
      ADSTACK_BLOCK_SIZE=4194304   
)

CB_ADD_GBENCHMARK(VectorModeComparison VectorModeComparison.cpp)
CB_ADD_GBENCHMARK(MemoryComplexity_tapenade MemoryComplexity.cpp)
CB_ADD_GBENCHMARK(Multithreading Multithreading.cpp)
CB_ADD_GBENCHMARK(Hessians Hessians.cpp)
CB_ADD_GBENCHMARK(GPT2Training GPT2Training.cpp)

if(APPLE)
  # On macOS, we want to explicitly use the high-performance Accelerate framework for BLAS.
  # We also need to give CMake a hint to find the Homebrew OpenMP library.
  list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp")
  find_package(OpenMP REQUIRED)
  # Link explicitly against Accelerate and the found OpenMP library.
  target_link_libraries(GPT2Training PRIVATE "-framework Accelerate" OpenMP::OpenMP_CXX)
else()
  # On all other platforms (e.g., Linux), find the best available BLAS and OpenMP.
  find_package(BLAS REQUIRED)
  find_package(OpenMP REQUIRED)
  # find_package(MKL CONFIG REQUIRED)
  target_link_libraries(GPT2Training PRIVATE ${BLAS_LIBRARIES} "-fopenmp=libomp" OpenMP::OpenMP_CXX)
  target_compile_options(GPT2Training PRIVATE -I/usr/include/mkl)
endif()
target_compile_options(GPT2Training PRIVATE -O3 -ffast-math)
target_compile_definitions(GPT2Training PRIVATE OMP)
if (BLAS_FOUND)
    target_compile_definitions(GPT2Training PRIVATE HAVE_CBLAS)
endif()

target_link_libraries(MemoryComplexity_tapenade PRIVATE tapenade_support)

add_custom_target(benchmark-clad COMMAND ${CMAKE_CTEST_COMMAND} -V
    DEPENDS ${CLAD_BENCHMARK_DEPS} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(benchmark-clad PROPERTIES FOLDER "Clad benchmarks")  

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  execute_process(
  COMMAND git apply --check "${CMAKE_CURRENT_SOURCE_DIR}/LULESH_diff.patch"
  WORKING_DIRECTORY "${lulesh-cuda_SOURCE_DIR}/LULESH"
  RESULT_VARIABLE patch_applied_or_not
  )

  if(patch_applied_or_not EQUAL 0)
    execute_process(
      COMMAND git apply "${CMAKE_CURRENT_SOURCE_DIR}/LULESH_diff.patch"
      WORKING_DIRECTORY "${lulesh-cuda_SOURCE_DIR}/LULESH"
    )
  endif()  

  set(LULESH_SOURCES
    ${lulesh-cuda_SOURCE_DIR}/LULESH/allocator.cu
    ${lulesh-cuda_SOURCE_DIR}/LULESH/lulesh.cu
    ${lulesh-cuda_SOURCE_DIR}/LULESH/lulesh-comms.cu
    ${lulesh-cuda_SOURCE_DIR}/LULESH/lulesh-comms-gpu.cu
  )
  
  set_source_files_properties(${LULESH_SOURCES} PROPERTIES LANGUAGE CXX)
  CB_ADD_CUDA_BENCHMARK(LULESH_Benchmark ${LULESH_SOURCES})
  target_compile_options(LULESH_Benchmark PRIVATE -fno-openmp)
  target_compile_definitions(LULESH_Benchmark PRIVATE 
  VERIFY=1 
  Normal_forward=0
  USE_CLAD=1
  )
  
  target_include_directories(LULESH_Benchmark PRIVATE ${lulesh-cuda_SOURCE_DIR}/LULESH) 
endif()


CB_ADD_GBENCHMARK(ActivityWaste ActivityWaste.cpp)

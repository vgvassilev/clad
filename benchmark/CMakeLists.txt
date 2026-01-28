set(CTEST_BUILD_NAME ${ROOT_ARCHITECTURE}-${CMAKE_BUILD_TYPE})
enable_testing()

set(CMAKE_CXX_COMPILER ${LLVM_TOOLS_BINARY_DIR}/clang++)
function(diff_compiler_test target)
  target_compile_options(
    ${target} PRIVATE  -Wno-undefined-inline)

  target_compile_definitions(${target} PRIVATE CLAD_TESTING=1)
endfunction()

cb_add_gbenchmark(Simple Simple.cpp)
cb_add_gbenchmark(AlgorithmicComplexity AlgorithmicComplexity.cpp)
cb_add_gbenchmark(ArrayExpressionTemplates ArrayExpressionTemplates.cpp)
if(CLAD_ENABLE_ENZYME_BACKEND)
  cb_add_gbenchmark(EnzymeCladComparison EnzymeCladComparison.cpp)
endif(CLAD_ENABLE_ENZYME_BACKEND)
cb_add_gbenchmark(VectorModeComparison VectorModeComparison.cpp)
cb_add_gbenchmark(MemoryComplexity MemoryComplexity.cpp)
cb_add_gbenchmark(Multithreading Multithreading.cpp)
cb_add_gbenchmark(Hessians Hessians.cpp)

set(BENCHMARK_TARGETS
    Simple
    AlgorithmicComplexity
    ArrayExpressionTemplates
    VectorModeComparison
    MemoryComplexity
    Multithreading
    Hessians)
foreach(bench ${BENCHMARK_TARGETS})
  diff_compiler_test(${bench})
endforeach()

set(CLAD_BENCHMARK_DEPS clad)
get_property(
  _benchmark_names
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  PROPERTY TESTS)

foreach(name ${_benchmark_names})
  get_test_property(${name} LABELS _labels)
  if(_labels MATCHES ".*benchmark.*")
    get_test_property(${name} DEPENDS _deps)
    list(APPEND CLAD_BENCHMARK_DEPS ${_deps})
  endif()
endforeach()

add_custom_target(
  benchmark-clad
  COMMAND ${CMAKE_CTEST_COMMAND} -V
  DEPENDS ${CLAD_BENCHMARK_DEPS}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

set_target_properties(benchmark-clad PROPERTIES FOLDER "Clad benchmarks")

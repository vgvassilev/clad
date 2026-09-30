# Unlike runtime unit tests, these drive Clang's AST/Sema APIs directly and do
# not compile the test itself through Clad's differentiation plugin.
add_unittest(CladUnitTests PullbackAdapterTests PullbackAdapter.cpp)
# The parent deliberately uses the clang C driver for all unit-test targets.
# Select its C++ link mode instead of hard-coding libstdc++ (wrong for libc++).
set_property(TARGET PullbackAdapterTests APPEND_STRING PROPERTY LINK_FLAGS
  " --driver-mode=g++")
get_target_property(engine_test_libs PullbackAdapterTests LINK_LIBRARIES)
list(REMOVE_ITEM engine_test_libs llvm_gtest_main llvm_gtest)
set_property(TARGET PullbackAdapterTests PROPERTY LINK_LIBRARIES "${engine_test_libs}")
find_package(Threads REQUIRED)
# Match the runtime-test helper's Homebrew libc++ search path.
if(APPLE AND LLVM_LIBRARY_DIR)
  target_link_directories(PullbackAdapterTests PRIVATE ${LLVM_LIBRARY_DIR}/c++)
endif()
target_link_libraries(PullbackAdapterTests PRIVATE
  gtest gtest_main cladPlugin cladDifferentiator Threads::Threads)
# Some packaged Clang installations export only clang-cpp, whereas in-tree
# builds expose the individual component targets. Do not turn missing targets
# into bare -lclangTooling flags, which silently defer the error until linking.
if(TARGET clangTooling)
  target_link_libraries(PullbackAdapterTests PRIVATE
    clangTooling clangFrontend clangSerialization clangSema clangAST clangLex
    clangBasic)
elseif(TARGET clang-cpp)
  target_link_libraries(PullbackAdapterTests PRIVATE clang-cpp)
else()
  message(FATAL_ERROR "Engine unit tests require Clang Tooling or clang-cpp")
endif()

# The surrounding test directory compiles with Clang even when the engine was
# built with GCC. Its instrumented archives need that GCC's gcov runtime, not
# the runtime from whichever GCC installation Clang happens to detect.
if(CLAD_CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  execute_process(COMMAND "${stored_cxx_compiler}" -print-file-name=libgcov.a
    OUTPUT_VARIABLE engine_gcov_runtime OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE engine_gcov_result)
  if(NOT engine_gcov_result EQUAL 0 OR NOT EXISTS "${engine_gcov_runtime}")
    message(FATAL_ERROR "Could not locate the engine compiler gcov runtime")
  endif()
  target_link_libraries(PullbackAdapterTests PRIVATE "${engine_gcov_runtime}")
endif()

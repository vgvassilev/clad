add_clad_unittest(AnalysisTests AnalysisBaseTest.cpp)

target_include_directories(AnalysisTests PRIVATE
  ${CLAD_SOURCE_DIR}/lib/Differentiator)

# Recipe installations can export clang-cpp without clangTooling.
if (TARGET clang-cpp)
  target_link_libraries(AnalysisTests PRIVATE cladDifferentiator clang-cpp)
elseif (TARGET clangTooling)
  target_link_libraries(AnalysisTests PRIVATE cladDifferentiator clangTooling)
else()
  message(FATAL_ERROR
    "AnalysisTests requires the clang-cpp or clangTooling CMake target")
endif()

if (CLAD_CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # These tests link instrumented clad objects, unlike plugin-only tests.
  # LINK_FLAGS also works with the project's older CMake minimum.
  set_property(TARGET AnalysisTests APPEND_STRING PROPERTY LINK_FLAGS
    " --coverage")

  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # The root build switches the test driver to Clang. Use the coverage
    # runtime belonging to the GCC that compiled cladDifferentiator.
    if (NOT stored_cxx_compiler)
      message(FATAL_ERROR
        "Cannot identify the GCC compiler used for cladDifferentiator")
    endif()
    execute_process(
      COMMAND "${stored_cxx_compiler}" -print-file-name=libgcov.a
      RESULT_VARIABLE _clad_gcov_result
      OUTPUT_VARIABLE _clad_gcov_library
      ERROR_VARIABLE _clad_gcov_error
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (NOT _clad_gcov_result EQUAL 0 OR
        NOT IS_ABSOLUTE "${_clad_gcov_library}" OR
        NOT EXISTS "${_clad_gcov_library}")
      message(FATAL_ERROR
        "Cannot locate GCC coverage runtime: ${_clad_gcov_error} "
        "${_clad_gcov_library}")
    endif()
    target_link_libraries(AnalysisTests PRIVATE "${_clad_gcov_library}")
  endif()
endif()

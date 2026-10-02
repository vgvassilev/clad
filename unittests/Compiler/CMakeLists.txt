add_clad_unittest(CompilerInstanceTests CompilerInstances.cpp)
target_link_libraries(CompilerInstanceTests PRIVATE clangTooling)
target_compile_definitions(CompilerInstanceTests PRIVATE
  CLAD_TEST_PLUGIN="$<TARGET_FILE:clad>"
  CLAD_TEST_INCLUDE="${CLAD_SOURCE_DIR}/include"
  CLAD_TEST_GENERATED_INCLUDE="${CLAD_BINARY_DIR}/include")

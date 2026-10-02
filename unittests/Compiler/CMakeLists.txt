add_clad_unittest(CompilerInstanceTests CompilerInstances.cpp)
set_target_properties(CompilerInstanceTests PROPERTIES ENABLE_EXPORTS ON)
target_link_libraries(CompilerInstanceTests PRIVATE clangTooling)
target_compile_definitions(CompilerInstanceTests PRIVATE
  "CLAD_TEST_PLUGIN=R\"($<TARGET_FILE:clad>)\""
  "CLAD_TEST_INCLUDE=R\"(${CLAD_SOURCE_DIR}/include)\""
  "CLAD_TEST_GENERATED_INCLUDE=R\"(${CLAD_BINARY_DIR}/include)\"")

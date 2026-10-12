add_clad_unittest(CompilerInstanceTests CompilerInstances.cpp)
# The plugin resolves clang's symbols in its host, so link clang the way the
# clang binary does. Where that is libclang-cpp, static clang libraries would
# carry only what this test uses, leaving the plugin's other symbols
# unresolved, and a plugin registry of their own besides.
if (CLANG_LINK_CLANG_DYLIB AND TARGET clang-cpp)
  target_link_libraries(CompilerInstanceTests PRIVATE clang-cpp)
else()
  target_link_libraries(CompilerInstanceTests PRIVATE clangTooling)
endif()
export_executable_symbols(CompilerInstanceTests)
target_compile_definitions(CompilerInstanceTests PRIVATE
  "CLAD_TEST_PLUGIN=R\"($<TARGET_FILE:clad>)\""
  "CLAD_TEST_INCLUDE=R\"(${CLAD_SOURCE_DIR}/include)\""
  "CLAD_TEST_GENERATED_INCLUDE=R\"(${CLAD_BINARY_DIR}/include)\"")

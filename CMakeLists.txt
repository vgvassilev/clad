set(MODULE TRUE)

set( LLVM_LINK_COMPONENTS support mc)

add_clang_library(AutoDiff AutoDiff.cpp)

add_dependencies(AutoDiff
  ClangAttrClasses
  ClangAttrList
  ClangCommentNodes
  ClangDeclNodes
  ClangDiagnosticCommon
  ClangStmtNodes
  )

target_link_libraries(AutoDiff
  clangFrontend
  clangAST
  )

set_target_properties(AutoDiff
  PROPERTIES
  LINKER_LANGUAGE CXX
  PREFIX "")

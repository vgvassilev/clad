set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  asmparser
  bitreader
  bitwriter
  codegen
  linker
  support
)

#set(LLVM_USED_LIBS
#  cladWhatever
#)


add_clad_library(cladDifferentiator
  ConstantFolder.cpp
  DerivativeBuilder.cpp
  DiffPlanner.cpp
  StmtClone.cpp
  Version.cpp
  )

add_dependencies(cladDifferentiator clangDriver ClangAttrList ClangDiagnosticDriver
                 ClangDriverOptions ClangCC1Options ClangCC1AsOptions)

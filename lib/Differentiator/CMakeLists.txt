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
#  autodiffWhatever
#)


add_autodiff_library(autodiffDifferentiator
  AutoDiff.cpp
  )

#add_dependencies(clangDriver ClangAttrList ClangDiagnosticDriver
#                 ClangDriverOptions ClangCC1Options ClangCC1AsOptions)

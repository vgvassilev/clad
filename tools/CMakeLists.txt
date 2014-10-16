if( NOT LLVM_BUILD_TOOLS )
  set(EXCLUDE_FROM_ALL ON)
endif()

add_llvm_loadable_module( clad
  ClangPlugin.cpp
  RequiredSymbols.cpp
  )

#add_dependencies(clad differentiator)

if( NOT CLAD_BUILT_STANDALONE AND NOT LLVM_BUILD_TOOLS )
  set(EXCLUDE_FROM_ALL ON)
endif()

add_llvm_library(cladPlugin
  ClangPlugin.cpp
  RequiredSymbols.cpp
  )

if (TARGET clangBasic)
  add_dependencies(cladPlugin clangBasic)
endif()

if (NOT CLAD_BUILD_STATIC_ONLY)
  add_llvm_loadable_module(clad
    ClangPlugin.cpp
    RequiredSymbols.cpp
    PLUGIN_TOOL
    clang
  )

  target_link_libraries(clad PRIVATE
    cladDifferentiator
  )
endif()

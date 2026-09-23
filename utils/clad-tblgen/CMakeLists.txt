# Built only where LLVM ships its TableGen library. What this tool renders is
# committed, so a build without it lacks nothing but the ability to regenerate.
set(LLVM_LINK_COMPONENTS Support)

add_llvm_executable(clad-tblgen TableGen.cpp)
# libLLVM carries no TableGen, so link the static library the top level found.
target_link_libraries(clad-tblgen PRIVATE ${CLAD_LLVM_TABLEGEN_LIB})
set_target_properties(clad-tblgen PROPERTIES FOLDER "Clad tools")

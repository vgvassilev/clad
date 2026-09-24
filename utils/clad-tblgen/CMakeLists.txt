# Built only where LLVM ships its TableGen library. What this tool renders is
# committed, so a build without it lacks nothing but the ability to regenerate.

# Let LLVM say how it wants to be linked; the answer decides which libraries
# appear and in what order.
#
# Without a libLLVM the components are archives, and naming TableGen is enough:
# LLVMExports gives it an INTERFACE_LINK_LIBRARIES of LLVMSupport, so cmake
# brings Support along and puts it after -- which is what a linker walking
# archives forward needs, since TableGen's references are satisfied by what
# follows it.
#
# With a libLLVM, llvm_config folds every component into it, and that library
# carries no TableGenMain. Name the one the top level found instead. It has to
# be the file rather than the imported target: the target would bring its
# include directories too, and on a Homebrew prefix that puts /usr/local
# ahead of the SDK's own headers. Nothing is lost by it -- a shared library
# leaves no undefined symbol for an archive to satisfy, so the order that
# matters above does not arise here.
if (LLVM_LINK_LLVM_DYLIB)
  set(LLVM_LINK_COMPONENTS Support)
  add_llvm_executable(clad-tblgen TableGen.cpp)
  target_link_libraries(clad-tblgen PRIVATE ${CLAD_LLVM_TABLEGEN_LIB})
else()
  set(LLVM_LINK_COMPONENTS TableGen)
  add_llvm_executable(clad-tblgen TableGen.cpp)
endif()
set_target_properties(clad-tblgen PROPERTIES FOLDER "Clad tools")

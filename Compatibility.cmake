## Compatibility

# Clang 8 remove add_llvm_loadable_module for cmake files.
# Recomended is use of add_llvm_library with MODULE argument.
if (NOT COMMAND add_llvm_loadable_module)
  macro(add_llvm_loadable_module module_name)
    add_llvm_library(${module_name} MODULE ${ARGN})
  endmacro(add_llvm_loadable_module)
endif()

# Clang 9 remove add_version_info_from_vcs.
# Recomended is use of get_source_info
if (NOT COMMAND add_version_info_from_vcs)
  set(clad_compat__isClang9 TRUE)
  macro(add_version_info_from_vcs VERS)
    get_source_info(${CMAKE_CURRENT_SOURCE_DIR} VERS DUMMY_REP)
  endmacro(add_version_info_from_vcs)
endif()

# Clang 9 change find_first_existing_vc_file interface.
# find_first_existing_vc_file(var path) --> find_first_existing_vc_file(path var)
if (clad_compat__isClang9)
  macro(clad_compat__find_first_existing_vc_file path out_var)
    find_first_existing_vc_file(${path} out_var)
  endmacro(clad_compat__find_first_existing_vc_file)
else()
  macro(clad_compat__find_first_existing_vc_file path out_var)
    find_first_existing_vc_file(out_var ${path})
  endmacro(clad_compat__find_first_existing_vc_file)
endif()

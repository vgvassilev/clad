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
  macro(add_version_info_from_vcs VERS)
    get_source_info(${CMAKE_CURRENT_SOURCE_DIR} VERS DUMMY_REP)
  endmacro(add_version_info_from_vcs)

# Clang 9 change find_first_existing_vc_file interface. (Clang => 9)
# find_first_existing_vc_file(var path) --> find_first_existing_vc_file(path var)
  macro(clad_compat__find_first_existing_vc_file path out_var)
    if (COMMAND find_first_existing_vc_file)
      find_first_existing_vc_file(${path} out_var)
    endif()
  endmacro(clad_compat__find_first_existing_vc_file)
else()
# Clang 9 change find_first_existing_vc_file interface. (Clang < 9)
# find_first_existing_vc_file(var path) --> find_first_existing_vc_file(path var)
  macro(clad_compat__find_first_existing_vc_file path out_var)
    if (COMMAND find_first_existing_vc_file)
      find_first_existing_vc_file(out_var ${path})
    endif()
  endmacro(clad_compat__find_first_existing_vc_file)
endif()

# Clang 9 VCS
function (clad_compat__DefineCustomCommandVCS version_inc LLVM_CMAKE_DIR CLANG_SOURCE_DIR CLAD_SOURCE_DIR clang_vc clad_vc)

  if (LLVM_CMAKE_DIR STREQUAL "")
    set(LLVM_CMAKE_DIR "${CLAD_SOURCE_DIR}/../../cmake/modules/")
  endif()

  set(generate_vcs_version_script "${LLVM_CMAKE_DIR}/GetSVN.cmake")
  if (NOT EXISTS ${generate_vcs_version_script})
    set(generate_vcs_version_script "${LLVM_CMAKE_DIR}/GenerateVersionFromVCS.cmake")
    # Create custom target to generate the VC revision include. (Clang >= 9)
    add_custom_command(OUTPUT "${version_inc}"
      DEPENDS "${generate_vcs_version_script}"
      COMMAND ${CMAKE_COMMAND} "-DNAMES=\"CLAD;CLANG\""
                           "-DCLAD_SOURCE_DIR=${CLAD_SOURCE_DIR}"
                           "-DCLANG_SOURCE_DIR=${CLANG_SOURCE_DIR}"
                           "-DLLVM_DIR=${LLVM_CMAKE_DIR}"
                           "-DCMAKE_MODULE_PATH=${LLVM_CMAKE_DIR}"
                           "-DHEADER_FILE=${version_inc}"
                           -P "${generate_vcs_version_script}")
  else()
    # Create custom target to generate the VC revision include. (Clang < 9)
        add_custom_command(OUTPUT "${version_inc}"
      DEPENDS "${clang_vc}" "${clad_vc}" "${get_svn_script}" "${generate_vcs_version_script}"
      COMMAND ${CMAKE_COMMAND} "-DFIRST_SOURCE_DIR=${CLANG_SOURCE_DIR}"
                           "-DFIRST_NAME=CLANG"
                           "-DSECOND_SOURCE_DIR=${CLAD_SOURCE_DIR}"
                           "-DSECOND_NAME=CLAD"
                           "-DLLVM_DIR=${LLVM_CMAKE_DIR}"
                           "-DCMAKE_MODULE_PATH=${LLVM_CMAKE_DIR}"
                           "-DHEADER_FILE=${version_inc}"
                           -P "${generate_vcs_version_script}")
  endif()
endfunction()

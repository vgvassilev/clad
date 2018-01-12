set(CLANG_SOURCE_DIR ${CLAD_SOURCE_DIR}/../clang/)
find_first_existing_vc_file(clang_vc "${CLANG_SOURCE_DIR}")
find_first_existing_vc_file(clad_vc "${CLAD_SOURCE_DIR}")
set(last_known_good_rev "${CLAD_SOURCE_DIR}/LastKnownGoodLLVMRevision.txt")

# The VC revision include that we want to generate.
set(version_inc "${CMAKE_CURRENT_BINARY_DIR}/SVNVersion.inc")

set(get_svn_script "${LLVM_CMAKE_PATH}/GetSVN.cmake")

if(DEFINED clang_vc AND DEFINED clad_vc)
  # Create custom target to generate the VC revision include.
  add_custom_command(OUTPUT "${version_inc}"
    DEPENDS "${clang_vc}" "${clad_vc}" "${get_svn_script}" "${last_known_good_rev}"
    COMMAND
    ${CMAKE_COMMAND} "-DFIRST_SOURCE_DIR=${CLANG_SOURCE_DIR}"
                     "-DFIRST_NAME=CLANG"
                     "-DSECOND_SOURCE_DIR=${CLAD_SOURCE_DIR}"
                     "-DSECOND_NAME=CLAD"
                     "-DHEADER_FILE=${version_inc}"
                     -P "${get_svn_script}")

  # Mark the generated header as being generated.
  set_source_files_properties("${version_inc}"
    PROPERTIES GENERATED TRUE
               HEADER_FILE_ONLY TRUE)

  # Tell Version.cpp that it needs to build with -DHAVE_SVN_VERSION_INC.
  set_source_files_properties(Version.cpp
    PROPERTIES COMPILE_DEFINITIONS "HAVE_CLAD_VERSION_INC")
endif()

file(READ ${last_known_good_rev} CLAD_CLANG_COMPAT_REVISION)
# Trim spaces
string(STRIP ${CLAD_CLANG_COMPAT_REVISION} CLAD_CLANG_COMPAT_REVISION)
set_property(SOURCE Version.cpp APPEND_STRING PROPERTY COMPILE_DEFINITIONS
  " -DCLAD_CLANG_COMPAT_REVISION=\"${CLAD_CLANG_COMPAT_REVISION}\" ")

# (Ab)use clang facilities for adding libraries.
set(LLVM_LINK_COMPONENTS support)
add_clang_library(cladDifferentiator
  ConstantFolder.cpp
  DerivativeBuilder.cpp
  DiffPlanner.cpp
  StmtClone.cpp
  Version.cpp
  ${version_inc}
  )

target_link_libraries(cladDifferentiator PRIVATE
    clangAST
    clangBasic
    clangSema
  )

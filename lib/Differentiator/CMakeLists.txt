set(CLANG_SOURCE_DIR ${CLAD_SOURCE_DIR}/../clang/)
clad_compat__find_first_existing_vc_file("${CLANG_SOURCE_DIR}" clang_vc)
clad_compat__find_first_existing_vc_file("${CLAD_SOURCE_DIR}" clad_vc)
set(last_known_good_rev "${CLAD_SOURCE_DIR}/LastKnownGoodLLVMRevision.txt")

# The VC revision include that we want to generate.
set(version_inc "${CMAKE_CURRENT_BINARY_DIR}/VCSVersion.inc")

set(generate_vcs_version_script "${LLVM_CMAKE_PATH}/GenerateVersionFromVCS.cmake")

# Create custom target to generate the VC revision include.
add_custom_command(OUTPUT "${version_inc}"
  DEPENDS "${clang_vc}" "${clad_vc}" "${generate_vcs_version_script}"
  COMMAND ${CMAKE_COMMAND} "-DNAMES=\"LLVM;CLANG\""
                           "-DLLVM_SOURCE_DIR=${llvm_source_dir}"
                           "-DCLANG_SOURCE_DIR=${clang_source_dir}"
                           "-DHEADER_FILE=${version_inc}"
                           -P "${generate_vcs_version_script}")

# Mark the generated header as being generated.
set_source_files_properties("${version_inc}"
  PROPERTIES GENERATED TRUE
             HEADER_FILE_ONLY TRUE)

set_property(SOURCE Version.cpp APPEND PROPERTY
             COMPILE_DEFINITIONS "HAVE_VCS_VERSION_INC")

file(READ ${last_known_good_rev} CLAD_CLANG_COMPAT_REVISION)
# Trim spaces
string(STRIP ${CLAD_CLANG_COMPAT_REVISION} CLAD_CLANG_COMPAT_REVISION)
# Apparantly, we cannot have multiple set_source_files_properties calls for the
# same file because it overrides older definitions.
set_property(SOURCE Version.cpp APPEND_STRING PROPERTY COMPILE_DEFINITIONS
  "CLAD_CLANG_COMPAT_REVISION=\"${CLAD_CLANG_COMPAT_REVISION}\"")

# (Ab)use llvm facilities for adding libraries.
add_llvm_library(cladDifferentiator
  ConstantFolder.cpp
  DerivativeBuilder.cpp
  DiffPlanner.cpp
  StmtClone.cpp
  Version.cpp
  ${version_inc}
  )

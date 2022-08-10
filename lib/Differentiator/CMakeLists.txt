set(CLANG_SOURCE_DIR ${CLAD_SOURCE_DIR}/../clang/)
if (NOT EXISTS ${CLANG_SOURCE_DIR})
  set(CLANG_SOURCE_DIR "${Clang_DIR}/")
endif()

clad_compat__find_first_existing_vc_file("${CLANG_SOURCE_DIR}" clang_vc)
clad_compat__find_first_existing_vc_file("${CLAD_SOURCE_DIR}" clad_vc)
set(last_known_good_rev "${CLAD_SOURCE_DIR}/LastKnownGoodLLVMRevision.txt")

# The VC revision include that we want to generate.
set(version_inc "${CMAKE_CURRENT_BINARY_DIR}/VCSVersion.inc")
clad_compat__DefineCustomCommandVCS("${version_inc}" "${LLVM_CMAKE_DIR}" "${CLANG_SOURCE_DIR}" "${CLAD_SOURCE_DIR}" "${clang_vc}" "${cald_vc}")

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
  CladUtils.cpp
  ConstantFolder.cpp
  DerivativeBuilder.cpp
  DiffPlanner.cpp
  ForwardModeVisitor.cpp
  HessianModeVisitor.cpp
  JacobianModeVisitor.cpp
  MultiplexExternalRMVSource.cpp
  ReverseModeVisitor.cpp
  ErrorEstimator.cpp
  EstimationModel.cpp
  StmtClone.cpp
  Version.cpp
  VisitorBase.cpp
  ${version_inc}
  )

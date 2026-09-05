# The VC revision include that we want to generate. Note that CLAD_SOURCE_DIR
# and CLAD_BINARY_DIR are ours also when clad is only a part of a bigger build,
# whereas CMAKE_SOURCE_DIR and CMAKE_BINARY_DIR belong to whoever is on top.
set(version_file ${CLAD_SOURCE_DIR}/VERSION)
set(version_inc ${CMAKE_CURRENT_BINARY_DIR}/VCSVersion.inc)
set(generate_vcs_version_script ${LLVM_CMAKE_DIR}/GenerateVersionFromVCS.cmake)
find_first_existing_vc_file(${CLAD_SOURCE_DIR} clad_vc)

# Create the VCSVersion.inc
add_custom_command(OUTPUT "${version_inc}"
  DEPENDS ${clad_vc} ${generate_vcs_version_script}
  COMMAND ${CMAKE_COMMAND} "-DNAMES=\"CLAD\""
                           "-DCLAD_SOURCE_DIR=${CLAD_SOURCE_DIR}"
                           "-DLLVM_DIR=${LLVM_CMAKE_DIR}"
                           "-DCMAKE_MODULE_PATH=${LLVM_CMAKE_DIR}"
                           "-DHEADER_FILE=${version_inc}"
                           -P "${generate_vcs_version_script}")


# Mark the generated header as being generated.
set_source_files_properties("${version_inc}"
  PROPERTIES GENERATED TRUE
             HEADER_FILE_ONLY TRUE)

set_property(SOURCE Version.cpp APPEND PROPERTY
  COMPILE_DEFINITIONS "HAVE_VCS_VERSION_INC")

# Configure the Version.inc file.
file(STRINGS ${version_file} CLAD_VERSION)
string(REPLACE "." ";" VERSION_LIST "${CLAD_VERSION}")
list(GET VERSION_LIST 0 CLAD_VERSION_MAJOR)
list(GET VERSION_LIST 1 CLAD_VERSION_MINOR)
list(GET VERSION_LIST 2 CLAD_VERSION_PATCH)

configure_file(
  ${CLAD_SOURCE_DIR}/include/clad/Differentiator/Version.inc.in
  ${CLAD_BINARY_DIR}/include/clad/Differentiator/Version.inc)


# (Ab)use llvm facilities for adding libraries.
llvm_add_library(cladDifferentiator
  STATIC
  ASTIntegrity.cpp
  ActivityAnalyzer.cpp
  AnalysisBase.cpp
  BaseForwardModeVisitor.cpp
  BaseForwardModeVisitorOpenMP.cpp
  CladUtils.cpp
  ConstantFolder.cpp
  DerivativeBuilder.cpp
  DerivedFnCollector.cpp
  DerivedFnInfo.cpp
  DiffPlanner.cpp
  ErrorEstimator.cpp
  DerivativePrinter.cpp
  GeneratedCode.cpp
  JacobianModeVisitor.cpp
  HessianModeVisitor.cpp
  MultiplexExternalRMVSource.cpp
  PushForwardModeVisitor.cpp
  ReverseModeForwPassVisitor.cpp
  ReverseModeVisitor.cpp
  ReverseModeVisitorCUDA.cpp
  ReverseModeVisitorEnzyme.cpp
  ReverseModeVisitorOpenMP.cpp
  TBRAnalyzer.cpp
  Timers.cpp
  StmtClone.cpp
  UsefulAnalyzer.cpp
  VectorForwardModeVisitor.cpp
  VectorPushForwardModeVisitor.cpp
  Version.cpp
  VisitorBase.cpp
  ${version_inc}
  )

# Version.cpp includes the generated VCSVersion.inc by its bare name, so the
# directory it is generated into has to be on the include path. Do not rely on
# CMAKE_INCLUDE_CURRENT_DIR here: a host project that adds us with
# add_subdirectory() need not have it set.
target_include_directories(cladDifferentiator PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

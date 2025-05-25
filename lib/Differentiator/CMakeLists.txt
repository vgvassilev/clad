# The VC revision include that we want to generate.
set(version_file ${CMAKE_SOURCE_DIR}/VERSION)
set(version_inc ${CMAKE_CURRENT_BINARY_DIR}/VCSVersion.inc)
set(generate_vcs_version_script ${LLVM_CMAKE_DIR}/GenerateVersionFromVCS.cmake)
find_first_existing_vc_file(${CMAKE_SOURCE_DIR} clad_vc)

# Create the VCSVersion.inc
add_custom_command(OUTPUT "${version_inc}"
  DEPENDS ${clad_vc} ${generate_vcs_version_script}
  COMMAND ${CMAKE_COMMAND} "-DNAMES=\"CLAD\""
                           "-DCLAD_SOURCE_DIR=${CMAKE_SOURCE_DIR}"
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
  ${CMAKE_SOURCE_DIR}/include/clad/Differentiator/Version.inc.in
  ${CMAKE_BINARY_DIR}/include/clad/Differentiator/Version.inc)


# (Ab)use llvm facilities for adding libraries.
llvm_add_library(cladDifferentiator
  STATIC
  ActivityAnalyzer.cpp
  BaseForwardModeVisitor.cpp
  CladUtils.cpp
  ConstantFolder.cpp
  DerivativeBuilder.cpp
  DerivedFnCollector.cpp
  DerivedFnInfo.cpp
  DiffPlanner.cpp
  ErrorEstimator.cpp
  EstimationModel.cpp
  JacobianModeVisitor.cpp
  HessianModeVisitor.cpp
  MultiplexExternalRMVSource.cpp
  PushForwardModeVisitor.cpp
  ReverseModeForwPassVisitor.cpp
  ReverseModeVisitor.cpp
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

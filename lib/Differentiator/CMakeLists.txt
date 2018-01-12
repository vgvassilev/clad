# Figure out if we can track VC revisions.
function(find_first_existing_file out_var)
  foreach(file ${ARGN})
    if(EXISTS "${file}")
      set(${out_var} "${file}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
endfunction()

macro(find_first_existing_vc_file out_var path)
  set(git_path "${path}/.git")

  # Normally '.git' is a directory that contains a 'logs/HEAD' file that
  # is updated as modifications are made to the repository. In case the
  # repository is a Git submodule, '.git' is a file that contains text that
  # indicates where the repository's Git directory exists.
  if (EXISTS "${git_path}" AND NOT IS_DIRECTORY "${git_path}")
    FILE(READ "${git_path}" file_contents)
    if("${file_contents}" MATCHES "^gitdir: ([^\n]+)")
      # '.git' is indeed a link to the submodule's Git directory.
      # Use the path to that Git directory.
      set(git_path "${path}/${CMAKE_MATCH_1}")
    endif()
  endif()

  find_first_existing_file(${out_var}
    "${git_path}/logs/HEAD"  # Git or Git submodule
    "${path}/.svn/wc.db"     # SVN 1.7
    "${path}/.svn/entries"   # SVN 1.6
    )
endmacro()

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
else()
  # Not producing a VC revision include.
  set(version_inc)

  # Being able to force-set the SVN revision in cases where it isn't available
  # is useful for performance tracking, and matches compatibility from autoconf.
  if(SVN_REVISION)
    set_source_files_properties(Version.cpp
       PROPERTIES COMPILE_DEFINITIONS "CLAD_REVISION=\"${SVN_REVISION}\"")
   endif()
endif()

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

target_link_libraries(cladDifferentiator PRIVATE
    clangAST
    clangBasic
    clangSema
  )

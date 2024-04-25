if( NOT CLAD_BUILT_STANDALONE AND NOT LLVM_BUILD_TOOLS )
  set(EXCLUDE_FROM_ALL ON)
endif()

set(CLAD_PLUGIN_SRC
  ClangPlugin.cpp
  ClangBackendPlugin.cpp
  RequiredSymbols.cpp
)

llvm_add_library(cladPlugin STATIC ${CLAD_PLUGIN_SRC})

set(link_libs cladDifferentiator)

if (NOT CLAD_BUILD_STATIC_ONLY)
  # Get rid of libLLVM-X.so which is appended to the list of static libraries.
  if (LLVM_LINK_LLVM_DYLIB)
    set(new_libs ${link_libs})
    set(libs ${new_libs})
    while(NOT "${new_libs}" STREQUAL "")
      foreach(lib ${new_libs})
        if(TARGET ${lib})
          get_target_property(transitive_libs ${lib} INTERFACE_LINK_LIBRARIES)
          if (NOT transitive_libs)
            continue()
          endif()
          foreach(transitive_lib ${transitive_libs})
            get_target_property(lib_type ${transitive_lib} TYPE)
            if("${lib_type}" STREQUAL "STATIC_LIBRARY")
              list(APPEND static_transitive_libs ${transitive_lib})
            else()
              # Filter our libLLVM.so and friends.
              continue()
            endif()
            if(NOT ${transitive_lib} IN_LIST libs)
              list(APPEND newer_libs ${transitive_lib})
              list(APPEND libs ${transitive_lib})
            endif()
          endforeach(transitive_lib)
          # Update the target properties with the list of only static libraries.
          set_target_properties(${lib} PROPERTIES INTERFACE_LINK_LIBRARIES "${static_transitive_libs}")
          set(static_transitive_libs "")
        endif()
      endforeach(lib)
      set(new_libs ${newer_libs})
      set(newer_libs "")
    endwhile()
  endif(LLVM_LINK_LLVM_DYLIB)

  add_llvm_loadable_module(
    clad ${CLAD_PLUGIN_SRC} DISABLE_LLVM_LINK_LLVM_DYLIB PLUGIN_TOOL clang
    )

  target_link_libraries(clad PRIVATE ${link_libs})

  # Add Enzyme as a backend.
  if (CLAD_ENABLE_ENZYME_BACKEND)

    if (LLVM_VERSION_MAJOR VERSION_LESS 9 OR LLVM_VERSION_MAJOR VERSION_GREATER 12)
      message(FATAL_ERROR "The enzyme backend is supported only in LLVM versions 9 to 12")
    endif()

    if (NOT UNIX)
      message(FATAL_ERROR "The enzyme backend is only supported on Unix platforms")
    endif()

    set(_enzyme_build_type ${CMAKE_CFG_INTDIR})
    set(_enzyme_cmake_logging_settings
      LOG_DOWNLOAD ON
      LOG_CONFIGURE ON
      LOG_BUILD ON
      LOG_INSTALL ON
      LOG_OUTPUT_ON_FAILURE ON
    )

    include(ExternalProject)

    set(_enzyme_patch_command
      ${CMAKE_COMMAND} -E copy_if_different
      ${CMAKE_SOURCE_DIR}/patches/enzyme.patch <SOURCE_DIR> &&
      git checkout <SOURCE_DIR> && git apply <SOURCE_DIR>/enzyme.patch
      )

    # Set build byproducts only needed by Ninja.
    set(_enzyme_static_target EnzymeStatic-${CLANG_VERSION_MAJOR})
    set(_enzyme_static_archive_name ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}${CMAKE_STATIC_LIBRARY_PREFIX}${_enzyme_static_target}${CMAKE_STATIC_LIBRARY_SUFFIX})
    if("${CMAKE_GENERATOR}" STREQUAL "Ninja")
      set(ENZYME_BYPRODUCTS ${_enzyme_static_archive_name})
    endif()

    ExternalProject_Add(
      Enzyme
      GIT_REPOSITORY https://github.com/wsmoses/Enzyme
      GIT_TAG v0.0.95
      GIT_SHALLOW 1 # Do not clone the history
      PATCH_COMMAND #${_enzyme_patch_command}
      UPDATE_COMMAND ""
      CMAKE_ARGS -G ${CMAKE_GENERATOR}
                 -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                 -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                 -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                 -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                 -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                 -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}
                 -DLLVM_DIR=${LLVM_DIR}
                 -DENZYME_CLANG=OFF
                 -DENZYME_STATIC_LIB=On
                 -DLLVM_EXTERNAL_LIT=${LLVM_EXTERNAL_LIT}
      SOURCE_SUBDIR enzyme
      BUILD_COMMAND ${CMAKE_COMMAND} --build . --config ${_enzyme_build_type} --target ${_enzyme_static_target} -- -j8
      #TEST_COMMAND ${CMAKE_COMMAND} --build . --config ${_enzyme_build_type} --target check-enzyme
      #TEST_BEFORE_INSTALL 1
      INSTALL_COMMAND ${CMAKE_COMMAND} --build . --config ${_enzyme_build_type} --target install-${_enzyme_static_target} -- -j8
      BUILD_BYPRODUCTS ${ENZYME_BYPRODUCTS}
      ${_enzyme_cmake_logging_settings}
      DEPENDS LLVMSupport
    )

    # Register LLVMEnzyme for our build system.
    add_library(LLVMEnzyme IMPORTED STATIC GLOBAL)
    add_dependencies(LLVMEnzyme Enzyme) # Emzyme is the ExternalProject_Add target.
    set_property(TARGET LLVMEnzyme PROPERTY IMPORTED_LOCATION ${_enzyme_static_archive_name})

    # Paste the contents of libLLVMEnzyme-X.so into clad.so.
    if (APPLE)
      set(_enzyme_link_flags -Wl,-force_load $<TARGET_FILE:LLVMEnzyme> -Wl,-force_load cladDifferentiator)
    elseif(MSVC)
      set(_enzyme_link_flags "-WHOLEARCHIVE:" $<TARGET_FILE:LLVMEnzyme>)
    else()
      set(_enzyme_link_flags -Wl,--whole-archive $<TARGET_FILE:LLVMEnzyme> -Wl,--no-whole-archive)
    endif()
    target_link_libraries(clad PUBLIC ${_enzyme_link_flags})

    add_dependencies(clad LLVMEnzyme)
  endif(CLAD_ENABLE_ENZYME_BACKEND)
endif()
set_source_files_properties(ClangPlugin.cpp PROPERTIES COMPILE_DEFINITIONS CLANG_MAJOR_VERSION="${CLANG_VERSION_MAJOR}")

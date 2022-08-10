add_library(cladCustomModelPlugin SHARED CustomModel.cpp)


if (TARGET check-clad)
  # Make sure if we are running the testsuite to build clingDemoPlugin before.
  add_dependencies(check-clad cladCustomModelPlugin)
endif()

# Sometimes it seems that even if we do not specify -frtti some compilers
# generate it by default. Then the library requires _ZTIN4clad22FPErrorEstimationModelE
# but it does not find it and fails to load.
target_compile_options(cladCustomModelPlugin PUBLIC -fno-rtti)

set_target_properties(cladCustomModelPlugin PROPERTIES  LIBRARY_OUTPUT_DIRECTORY ".")
if(APPLE)
  target_link_libraries(cladCustomModelPlugin PUBLIC -Wl,-bind_at_load -Wl,-undefined -Wl,dynamic_lookup)
elseif(NOT MSVC)
  target_link_libraries(cladCustomModelPlugin PUBLIC -Wl,--unresolved-symbols=ignore-in-object-files)
endif()

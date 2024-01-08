add_clad_unittest(KokkosTests
  parallel_for.cpp
  main.cpp
  )

target_link_libraries(KokkosTests PUBLIC ${Kokkos_LIBRARIES})
target_include_directories(KokkosTests SYSTEM PRIVATE ${Kokkos_INCLUDE_DIRS})

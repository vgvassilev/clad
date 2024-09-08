add_clad_unittest(KokkosTests
  main.cpp
  ViewAccess.cpp
  ViewBasics.cpp
  ParallelReduce.cpp
  ParallelFor.cpp
  )

# If llvm does not require rtti, kokkos does.
if (NOT (LLVM_REQUIRES_RTTI OR LLVM_ENABLE_RTTI))
  if (LLVM_COMPILER_IS_GCC_COMPATIBLE)
    target_compile_options(KokkosTests PUBLIC -frtti)
  elseif(MSVC)
    target_compile_options(KokkosTests PUBLIC /GR)
  endif()
endif()

target_link_libraries(KokkosTests PUBLIC ${Kokkos_LIBRARIES})
target_include_directories(KokkosTests SYSTEM PRIVATE ${Kokkos_INCLUDE_DIRS})

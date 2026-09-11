add_clad_unittest(KokkosTests
  main.cpp
  ViewAccess.cpp
  ViewBasics.cpp
  ParallelReduce.cpp
  ParallelFor.cpp
  ViewCtorForward.cpp
  )

# If llvm does not require rtti, kokkos does.
if (NOT (LLVM_REQUIRES_RTTI OR LLVM_ENABLE_RTTI))
  if (LLVM_COMPILER_IS_GCC_COMPATIBLE)
    target_compile_options(KokkosTests PUBLIC -frtti)
  elseif(MSVC)
    target_compile_options(KokkosTests PUBLIC /GR)
  endif()
endif()

# Link the imported target, not Kokkos_LIBRARIES/Kokkos_INCLUDE_DIRS: it
# carries the usage requirements, notably cxx_std_20, which Kokkos 5.x needs
# (its headers use std::type_identity). The variables drop them.
target_link_libraries(KokkosTests PUBLIC Kokkos::kokkos)

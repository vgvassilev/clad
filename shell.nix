# Declaration of an environment that can be used to build Clad and compile
# binaries with clang++ using the Clad plugin.
#
# Provided environment variables:
#   - CMAKE_FLAGS : flags for configuring with CMake

{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  nativeBuildInputs = with pkgs; [
    llvmPackages_16.clang-unwrapped # for Clang library
    llvmPackages_16.clangUseLLVM # for clang wrapper (useful to compile code that tests Cland)
    llvmPackages_16.libcxxStdenv # standard C++ library for Clang
    llvmPackages_16.stdenv # standard C library for Clang
    llvm_16 # using LLVM 16, because this is also what ROOT uses
  ];

  shellHook =
    with {
      cmakeFlags = [
        "-DCLAD_DISABLE_TESTS=ON"
        "-DLLVM_DIR=${pkgs.llvm_16.dev}"
        "-DClang_DIR=${pkgs.llvmPackages_16.clang-unwrapped.dev}"
      ];
    }; ''
      CMAKE_FLAGS="${pkgs.lib.strings.concatStrings (pkgs.lib.strings.intersperse " " cmakeFlags)}"
    '';
}

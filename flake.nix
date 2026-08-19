{
  description = "Development environment for clad, the Clang plugin for automatic differentiation";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      # Environment for building clad and running its test suite:
      #
      #   nix develop
      #   cmake -S . -B build $CMAKE_FLAGS
      #   cmake --build build -j$(nproc)
      #   cmake --build build --target check-clad
      #
      # Provided environment variables:
      #   - CMAKE_FLAGS : flags for configuring with CMake
      devShells = forAllSystems (
        pkgs:
        let
          # LLVM 22 is the newest release supported by clad, see
          # LLVM_MAX_SUPPORTED in CMakeLists.txt.
          llvmPackages = pkgs.llvmPackages_22;

          # Teach the clang wrapper about the two things the tests need from it.
          # This has to be baked into the wrapper rather than exported by the
          # shell, because lit runs the compiler with a stripped environment.
          clang = llvmPackages.clang.override (old: {
            extraBuildCommands = (old.extraBuildCommands or "") + ''
              # The wrapper passes linker flags to clang even when only
              # compiling, and the resulting "unused argument" warnings make the
              # tests fail, since they assert that clang stays silent.
              echo "-Qunused-arguments" >> $out/nix-support/cc-cflags

              # Make -fopenmp work out of the box, for the OpenMP tests.
              echo "-isystem ${llvmPackages.openmp.dev}/include" >> $out/nix-support/cc-cflags
              echo "-L${llvmPackages.openmp}/lib" >> $out/nix-support/cc-ldflags
            '';
          });

          # Nixpkgs installs clang outside of LLVM's own prefix, but clad
          # expects to find it next to the LLVM tools: it compiles the unit
          # tests (and googletest) with ${LLVM_TOOLS_BINARY_DIR}/clang, and lit
          # looks for clang and llvm-config there as well. Hence a joined prefix
          # containing all of them, with the *wrapped* clang so that it knows
          # where the C and C++ standard libraries live.
          llvmTools = pkgs.symlinkJoin {
            name = "llvm-tools-with-clang-${llvmPackages.llvm.version}";
            paths = [
              llvmPackages.llvm # FileCheck, not, count, ...
              llvmPackages.llvm.dev # llvm-config
              clang
            ];
          };

          # LLVMConfig.cmake hardcodes LLVM_TOOLS_BINARY_DIR, so re-point it at
          # the joined prefix above. Everything else keeps referring to the
          # original store paths.
          llvmCMakeDir = pkgs.runCommand "llvm-cmake-dir-${llvmPackages.llvm.version}" { } ''
            mkdir -p $out
            ln -s ${llvmPackages.llvm.dev}/lib/cmake/llvm/* $out/
            rm $out/LLVMConfig.cmake
            substitute ${llvmPackages.llvm.dev}/lib/cmake/llvm/LLVMConfig.cmake $out/LLVMConfig.cmake \
              --replace-fail "${llvmPackages.llvm}/bin" "${llvmTools}/bin"
          '';

          cmakeFlags = [
            "-DLLVM_DIR=${llvmCMakeDir}"
            "-DClang_DIR=${llvmPackages.clang-unwrapped.dev}/lib/cmake/clang"
            # LLVM releases don't ship lit and nixpkgs doesn't install llvm-lit.
            "-DLLVM_EXTERNAL_LIT=${pkgs.lit}/bin/lit"
          ];
        in
        {
          # Build clad with the very clang it is a plugin for.
          default = (pkgs.mkShell.override { stdenv = pkgs.overrideCC llvmPackages.stdenv clang; }) {
            packages = [
              pkgs.cmake
              pkgs.git # the unit tests fetch googletest with ExternalProject
              pkgs.lit
              llvmPackages.llvm # FileCheck & friends, to rerun tests by hand
            ];

            env.CMAKE_FLAGS = builtins.concatStringsSep " " cmakeFlags;
          };
        }
      );
    };
}

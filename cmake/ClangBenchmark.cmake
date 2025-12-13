function(clad_apply_benchmark_compiler target)
    target_compile_options(
    ${target} PRIVATE  -Wno-undefined-inline)

    set_property(TARGET ${target} PROPERTY CXX_COMPILER ${LLVM_TOOLS_BINARY_DIR}/clang++)
endfunction()
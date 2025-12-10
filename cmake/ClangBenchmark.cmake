function(clad_apply_test_compiler target)
    target_compile_options(${target} PRIVATE
        -Wno-class-memaccess
        -Wno-undefined-inline
        -fno-lifetime-dse
    )

    target_compile_options(${target} PRIVATE
        $<$<NOT:$<CXX_COMPILER_ID:Clang>>:-flags>
    )

    target_compile_definitions(${target} PRIVATE CLAD_TESTING=1)

    set_property(TARGET ${target} PROPERTY CXX_COMPILER ${LLVM_TOOLS_BINARY_DIR}/clang++)
endfunction()
# Clad Diagnostic Groups

Clad organizes its diagnostics into logical groups, allowing fine-grained control via `-W` flags:

- **`-Wclad`**: Main group (enables all by default)
- **`-Wclad-unsupported`**: Unsupported constructs (indirect calls, inline assembly, etc.)
- **`-Wclad-pragma`**: Pragma parsing errors
- **`-Wclad-checkpointing`**: Checkpointing directives
- **`-Wclad-builtin`**: Built-in function issues
- **`-Wclad-non-differentiable`**: Non-differentiable operations

## Usage

```bash
# Disable all clad diagnostics
clang -fplugin=libclad.so -Xclang -plugin-arg-clad -Xclang -Wno-clad file.cpp

# Disable unsupported construct warnings only
clang -fplugin=libclad.so -Xclang -plugin-arg-clad -Xclang -Wno-clad-unsupported file.cpp

# Disable pragma warnings only
clang -fplugin=libclad.so -Xclang -plugin-arg-clad -Xclang -Wno-clad-pragma file.cpp
```

All diagnostics are shown by default. This feature allows users to selectively suppress warning categories to match project policies.

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
clang -Wno-clad file.cpp -fplugin=libclad.so

# Enable only unsupported construct warnings
clang -Wno-clad -Wclad-unsupported file.cpp -fplugin=libclad.so

# Disable pragma warnings, keep others
clang -Wno-clad-pragma file.cpp -fplugin=libclad.so
```

All diagnostics are shown by default. This feature allows users to selectively suppress warning categories to match project policies.

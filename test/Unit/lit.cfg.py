# -*- Python -*-

import platform

import lit.formats

config.name = "Clad Unit Tests"
config.suffixes = []  # Seems not to matter for google tests?

# Test Source and Exec root dirs both point to the same directory where google
# test binaries are built.
config.test_exec_root = os.path.join(config.clad_obj_root, "unittests")
config.test_source_root = config.test_exec_root
print(config.test_exec_root)
# All GoogleTests are named to have 'Tests' as their suffix. The '.' coming from
# llvm_build_mode option is a special value for GoogleTest indicating that it
# should look through the entire testsuite recursively for tests (alternatively,
# one could provide a ;-separated list of subdirectories).
config.test_format = lit.formats.GoogleTest(config.llvm_build_mode, "Tests")

if platform.system() == "Darwin":
    shlibpath_var = "DYLD_LIBRARY_PATH"
elif platform.system() == "Windows":
    shlibpath_var = "PATH"
else:
    shlibpath_var = "LD_LIBRARY_PATH"

# Point the dynamic loader at dynamic libraries in 'lib'.
shlibpath = os.path.pathsep.join(
    (config.shlibdir, config.llvm_libs_dir, config.environment.get(shlibpath_var, ""))
)

# Win32 seeks DLLs along %PATH%.
if sys.platform in ["win32", "cygwin"] and os.path.isdir(config.shlibdir):
    shlibpath = os.path.pathsep.join((config.shlibdir, shlibpath))

config.environment[shlibpath_var] = shlibpath

# It is not realistically possible to account for all options that could
# possibly be present in system and user configuration files, so disable
# default configs for the test runs.
config.environment["CLANG_NO_DEFAULT_CONFIG"] = "1"

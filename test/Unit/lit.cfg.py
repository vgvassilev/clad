# -*- Python -*-

import platform

import lit.formats

# When lit is started on the source directory by check-clad-unit the
# generated site configuration has not been loaded yet. Because unit is a secondary
# generation site unlike others in tests/, the initial lit skips the configuration 
# in tests/Unit and because there is no fallback for config in Units/ 
# unlike in tests/ ,this skips the config. This check will run after top-level pass 
# and loads the local config needed for check-clad-unit 
if not hasattr(config, "clad_obj_root"):
    site_cfg = lit_config.params.get("clad_site_config")
    unit_site_cfg = (
        os.path.join(os.path.dirname(site_cfg), "Unit", "lit.site.cfg.py")
        if site_cfg
        else None
    )
    if not unit_site_cfg or not os.path.exists(unit_site_cfg):
        lit_config.fatal("No site specific configuration available for Unit tests!")
    lit_config.load_config(config, unit_site_cfg)
    raise SystemExit

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

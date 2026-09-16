#!/usr/bin/env python3
import sys
import subprocess
import os

def main():
    if len(sys.argv) < 3:
        print("Usage: compile_produced_code.py <clad_cmd> [compiler_flags...] <source_file>")
        sys.exit(1)

    cmd = sys.argv[1:]
    
    # Run the clad command to produce derived source code output
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception as e:
        print(f"Error executing clad command: {e}")
        sys.exit(1)

    output = proc.stdout + "\n" + proc.stderr
    
    # Extract produced derived code lines
    derived_lines = []
    recording = False
    for line in output.splitlines():
        if "generated-source:" in line or "derived-fn:" in line or "Derived Function:" in line:
            recording = True
            continue
        if recording:
            derived_lines.append(line)

    if not derived_lines:
        # Fallback: if no specific header marker, use full output
        derived_code = output
    else:
        derived_code = "\n".join(derived_lines)

    # Compile the produced code with clang++ -fsyntax-only
    compiler = os.environ.get("CLANG", "clang++")
    compile_cmd = [compiler, "-std=c++17", "-fsyntax-only", "-x", "c++", "-"]
    
    try:
        res = subprocess.run(compile_cmd, input=derived_code, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            print("Failed to compile produced source code:")
            print(res.stderr)
            sys.exit(res.returncode)
    except Exception as e:
        print(f"Error executing compiler: {e}")
        sys.exit(1)

    print("Produced code compiled successfully.")
    sys.exit(0)

if __name__ == "__main__":
    main()

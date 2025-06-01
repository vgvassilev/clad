import subprocess
import argparse
import os
from pathlib import Path

def run_cmd(cmd, check=True):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=check)

def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results between two Git revisions.")
    parser.add_argument("baseline_revision", help="The baseline Git revision")
    parser.add_argument("current_revision", help="The current Git revision to compare against the baseline")
    args = parser.parse_args()

    original_dir = os.getcwd()

    baseline_rev = subprocess.check_output(["git", "rev-parse", args.baseline_revision], text=True).strip()
    current_rev = subprocess.check_output(["git", "rev-parse", args.current_revision], text=True).strip()

    os.chdir("..")
    run_cmd(["git", "checkout", baseline_rev])
    os.chdir("build")
    run_cmd(["cmake", "--build", ".", "--target", "benchmark-clad", "-j4"])
    os.chdir("..")

    run_cmd(["git", "checkout", current_rev])
    os.chdir("build")
    run_cmd(["cmake", "--build", ".", "--target", "benchmark-clad", "-j4"])

    run_cmd(["pip3", "install", "-r", "./googlebenchmark-prefix/src/googlebenchmark/tools/requirements.txt"])
    comparer = str(Path.cwd() / "googlebenchmark-prefix/src/googlebenchmark/tools/compare.py")

    os.chdir("benchmark")
    for baseline_file in os.listdir():
        if baseline_file.endswith(f"{baseline_rev}.json"):
            common_prefix = baseline_file[: -len(f"{baseline_rev}.json")]
            for filename in os.listdir():
                if (filename.startswith(common_prefix) and filename.endswith(".json") and filename != baseline_file):
                    print(f"Running 'python3 {comparer} benchmarks {baseline_file} {filename}'")
                    run_cmd(["python3", comparer, "benchmarks", baseline_file, filename])
                    break

    os.chdir(original_dir)

if __name__ == "__main__":
    main()

"""一键运行所有实验"""
import sys
import os
import subprocess
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

EXPERIMENTS = [
    ("Exp 1: Convexity",          "scripts/run_exp1.py"),
    ("Exp 2: Asymptotic Scale",   "scripts/run_exp2.py"),
    ("Exp 3: Phase Transition",   "scripts/run_exp3.py"),
    ("Exp 4: Epsilon Effect",     "scripts/run_exp4.py"),
    ("Exp 5: Majority Scale",     "scripts/run_exp5.py"),
    ("Exp 6: Comparison",         "scripts/run_exp6.py"),
]


def run_script(path):
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  Running: {path}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, path], check=False)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"[{status}] {path}  ({elapsed:.1f}s)")
    return result.returncode


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None

    for name, path in EXPERIMENTS:
        if only is not None and only not in name.lower() and only not in path:
            continue
        code = run_script(path)
        if code != 0:
            print(f"\n[STOP] {path} failed with code {code}. Aborting.")
            sys.exit(code)

    print(f"\n{'='*60}")
    print("  All experiments finished successfully.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
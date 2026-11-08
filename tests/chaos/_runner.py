"""Subprocess entry: chdir to the checkout and run run_agent.main(); timing knobs come from HARNESS_* env."""
import os
import sys

if __name__ == "__main__":
    ROOT = os.environ["CHAOS_ROOT"]
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    import run_agent

    run_agent.main(sys.argv[1:])

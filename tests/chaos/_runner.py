"""Subprocess entry: apply test-only knobs (the product code has no env for them), then run_agent.main().

CHAOS_LLM_TIMEOUT, CHAOS_RETRY_DELAYS (comma list), CHAOS_DEADLINE_S, CHAOS_STOP_BEFORE_S,
CHAOS_PHASE1_TIMEOUT (default seconds).
"""
import os
import sys

def _setup():
    ROOT = os.environ["CHAOS_ROOT"]
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    from harness import config, llm, scheduler  # noqa: E402

    e = os.environ
    if e.get("CHAOS_LLM_TIMEOUT"):
        llm.LLMClient.__init__.__defaults__ = (llm.LLM_CONCURRENCY, float(e["CHAOS_LLM_TIMEOUT"]))
    if e.get("CHAOS_RETRY_DELAYS"):
        llm.RETRY_DELAYS = tuple(float(x) for x in e["CHAOS_RETRY_DELAYS"].split(","))
    if e.get("CHAOS_DEADLINE_S"):
        scheduler.run.__defaults__ = (float(e["CHAOS_DEADLINE_S"]),) + scheduler.run.__defaults__[1:]
    if e.get("CHAOS_STOP_BEFORE_S"):
        config.STOP_NEW_BEFORE_DEADLINE_S = float(e["CHAOS_STOP_BEFORE_S"])
    if e.get("CHAOS_PHASE1_TIMEOUT"):
        config.PHASE1_TIMEOUT["default"] = float(e["CHAOS_PHASE1_TIMEOUT"])



if __name__ == "__main__":
    _setup()
    import run_agent

    run_agent.main(sys.argv[1:])

"""Two-phase orchestration: free deterministic solvers first, then budgeted model routes."""
import asyncio
import logging
import multiprocessing
import os
import queue as queue_mod
import tempfile
import time
from pathlib import Path

from harness import config
from harness.budget import TokenLedger
from harness.estimator import estimate_json_tokens, estimate_tokens
from harness.result import SolveResult
from harness.tasks import Task, load_tasks, write_answer
from harness.trajectory import Trajectory

log = logging.getLogger("harness")

LLM_WORKERS = config.LLM_CONCURRENCY
JSON_TYPES = ("math", "data_analysis", "search")

# resolved lazily (importing them imports openjiuwen); tests may pre-set them
solve_group = None
solve_with_program = None
fix_coding = None
BudgetExhausted = None


def _log_dir() -> Path:
    return Path(os.environ.get("HARNESS_LOG_DIR") or Path(tempfile.gettempdir()) / "harness")


def _solver_for(task_type: str):
    from harness.solvers import coding_solver, data_solver, math_solver, search_solver
    return {"math": math_solver, "data_analysis": data_solver, "search": search_solver,
            "coding": coding_solver}.get(task_type)


def _phase1_target(out_queue, task: Task, solver_name=None) -> None:
    """Child process body: run one deterministic solver, send (task_id, SolveResult)."""
    try:
        mod = _solver_for(task.task_type)
        res = mod.solve(task) if mod is not None else SolveResult("unparsed", note="unknown task type")
    except BaseException as e:  # the child must always answer
        res = SolveResult("unparsed", note=f"solver crashed: {e!r}")
    out_queue.put((task.task_id, res))


def _kill(proc) -> None:
    try:
        proc.terminate()
        proc.join(1.0)
        if proc.is_alive():
            proc.kill()
            proc.join(1.0)
    except Exception:
        pass


def _phase1(tasks: list, workers: int, t0: float, deadline_s: float, on_result) -> None:
    ctx = multiprocessing.get_context("spawn")
    out_queue = ctx.Queue()
    pending = list(tasks)
    running: dict = {}  # task_id -> (proc, start, task)
    done: set = set()

    def finish(task: Task, res: SolveResult, start: float) -> None:
        done.add(task.task_id)
        try:
            on_result(task, res, time.monotonic() - start)
        except Exception:
            log.exception("phase 1: handling result of %s failed", task.task_id)

    def drain(timeout: float = 0.0) -> None:
        while True:
            try:
                tid, res = out_queue.get(timeout=timeout) if timeout else out_queue.get_nowait()
            except (queue_mod.Empty, OSError, EOFError):
                return
            ent = running.pop(tid, None)
            if ent is not None and tid not in done:
                proc, start, task = ent
                finish(task, res, start)
                proc.join(1.0)
                if proc.is_alive():
                    _kill(proc)
            timeout = 0.0

    try:
        while pending or running:
            now = time.monotonic()
            over = now - t0 > deadline_s
            while pending and len(running) < workers and not over:
                task = pending.pop(0)
                try:
                    proc = ctx.Process(target=_phase1_target, args=(out_queue, task, None), daemon=True)
                    proc.start()
                    running[task.task_id] = (proc, time.monotonic(), task)
                except Exception as e:
                    finish(task, SolveResult("unparsed", note=f"spawn failed: {e!r}"), now)
            if over:
                for task in pending:
                    finish(task, SolveResult("unparsed", note="deadline before phase 1 start"), now)
                pending.clear()
            drain()
            for tid, (proc, start, task) in list(running.items()):
                limit = config.PHASE1_TIMEOUT.get(task.task_type, config.PHASE1_TIMEOUT["default"])
                if not proc.is_alive():
                    drain(0.3)  # the result may still be in the pipe
                    if tid in running:
                        running.pop(tid)
                        finish(task, SolveResult("unparsed", note="solver process died"), start)
                elif time.monotonic() - start > limit or over:
                    _kill(proc)
                    drain()
                    if tid in running:
                        running.pop(tid)
                        finish(task, SolveResult("unparsed", note="phase 1 timeout"), start)
            if running or pending:
                time.sleep(0.02)
    finally:
        for proc, _, _ in running.values():
            _kill(proc)
        try:
            out_queue.close()
            out_queue.cancel_join_thread()
        except Exception:
            pass


def _load_phase2() -> None:
    global solve_group, solve_with_program, fix_coding, BudgetExhausted
    from harness import json_route, llm, llm_routes
    if solve_group is None:
        solve_group = json_route.solve_group
    if solve_with_program is None:
        solve_with_program = llm_routes.solve_with_program
    if fix_coding is None:
        fix_coding = llm_routes.fix_coding
    BudgetExhausted = llm.BudgetExhausted


async def _drive(units: list, t0: float, deadline_s: float, ledger: TokenLedger,
                 state: dict) -> None:
    """Run ordered units on LLM_WORKERS workers; stop issuing on exhaustion or near the deadline."""
    it = iter(units)

    async def worker() -> None:
        while not state["stop"]:
            if time.monotonic() - t0 > deadline_s - config.STOP_NEW_BEFORE_DEADLINE_S:
                state["stop"] = True
                return
            unit = next(it, None)
            if unit is None:
                return
            if not ledger.can_afford(unit["estimate"]):
                if ledger.exhausted:
                    state["stop"] = True
                    return
                continue
            start = time.monotonic()
            try:
                await unit["run"]()
            except BudgetExhausted:
                state["stop"] = True
                ledger.mark_exhausted()
                state["traj"].record({"phase": 2, "task_id": unit["name"], "status": "budget_exhausted",
                                      "cost": 0, "elapsed": round(time.monotonic() - start, 3)})
            except Exception as e:
                log.exception("unit %s failed", unit["name"])
                state["traj"].record({"phase": 2, "task_id": unit["name"], "status": "error",
                                      "cost": 0, "note": repr(e)[:200],
                                      "elapsed": round(time.monotonic() - start, 3)})

    await asyncio.gather(*(worker() for _ in range(max(1, LLM_WORKERS))))


def run(tasks_dir: Path, budget: float, client, deadline_s: float = config.DEADLINE_S,
        phase1_workers: int = config.PHASE1_WORKERS) -> dict:
    t0 = time.monotonic()
    traj = Trajectory(_log_dir())
    ledger = getattr(client, "ledger", None) or TokenLedger(budget, budget * config.RESERVE_FRACTION)
    status: dict = {}
    try:
        try:
            tasks = load_tasks(Path(tasks_dir))
        except Exception:
            log.exception("cannot load tasks from %s", tasks_dir)
            tasks = []
        by_id = {t.task_id: t for t in tasks}
        last_spent = [ledger.spent]

        def settle(phase: int, task: Task, res: SolveResult, elapsed: float) -> None:
            st = res.status
            if st == "solved" and res.answer is None:
                st = "unparsed"
            elif st == "solved":
                try:
                    write_answer(task, res.answer)
                except Exception as e:
                    log.exception("cannot write answer for %s", task.task_id)
                    st, res = "unparsed", SolveResult("unparsed", note=f"write failed: {e!r}")
            status[task.task_id] = st
            # cost deltas telescope, so their sum equals the ledger's spent exactly
            delta = ledger.spent - last_spent[0]
            last_spent[0] = ledger.spent
            traj.record({"phase": phase, "task_id": task.task_id, "status": st, "cost": delta,
                         "elapsed": round(elapsed, 3), "note": (res.note or "")[:200]})

        _phase1(tasks, phase1_workers, t0, deadline_s, lambda t, r, e: settle(1, t, r, e))

        left = [by_id[i] for i, s in status.items() if s == "unparsed"]
        if client is not None and left:
            _phase2(left, client, ledger, t0, deadline_s, settle, status, traj)
    except Exception:
        log.exception("scheduler failed")
    finally:
        traj.close()
    counts = {"solved": 0, "infeasible": 0, "unparsed": 0}
    for s in status.values():
        counts[s if s in counts else "unparsed"] += 1
    counts["spent"] = ledger.spent
    return counts


def _phase2(left, client, ledger, t0, deadline_s, settle, status, traj) -> None:
    try:
        os.chdir(_log_dir())  # openjiuwen writes ./logs relative to cwd
    except OSError:
        pass
    _load_phase2()
    from harness.templates import group
    state = {"stop": False, "traj": traj}

    # 2a: one JSON-spec call per template group
    units = []
    try:
        groups = group([t for t in left if t.task_type in JSON_TYPES])
    except Exception:
        log.exception("grouping failed")
        groups = {}
    for (ttype, template), members in groups.items():
        est = estimate_json_tokens(template)

        def make(ttype=ttype, template=template, members=members):
            async def go():
                start = time.monotonic()
                results = await solve_group(ttype, template, members, client)
                el = time.monotonic() - start
                for task, _ in members:
                    res = results.get(task.task_id)
                    if res is not None and res.status in ("solved", "infeasible"):
                        settle(2, task, res, el)
            return go
        reward = sum(t.reward for t, _ in members)
        units.append({"prio": reward / est, "estimate": est, "run": make(),
                      "name": f"group:{ttype}:{len(members)}"})
    units.sort(key=lambda u: -u["prio"])
    _run_async(_drive(units, t0, deadline_s, ledger, state))

    # 2b: per-task routes for whatever is still unparsed
    units = []
    for task in [t for t in left if status.get(t.task_id) == "unparsed"]:
        est = estimate_tokens(task)

        def make(task=task):
            async def go():
                start = time.monotonic()
                if task.task_type == "coding":
                    res = await fix_coding(task, client)
                else:
                    res = await solve_with_program(task, client)
                settle(2, task, res, time.monotonic() - start)
            return go
        units.append({"prio": task.reward / est, "estimate": est, "run": make(), "name": task.task_id})
    units.sort(key=lambda u: -u["prio"])
    _run_async(_drive(units, t0, deadline_s, ledger, state))


def _run_async(coro) -> None:
    try:
        asyncio.run(coro)
    except Exception:
        log.exception("phase 2 loop failed")

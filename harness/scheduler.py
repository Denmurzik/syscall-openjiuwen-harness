"""Two-phase orchestration: free deterministic solvers first, then budgeted model routes."""
import asyncio
import logging
import multiprocessing
import os
import queue as queue_mod
import tempfile
import time
from contextvars import ContextVar
from pathlib import Path

from harness import config
from harness.budget import TokenLedger
from harness.estimator import estimate_json_tokens, estimate_tokens
from harness.result import SolveResult
from harness.tasks import Task, load_tasks, write_answer
from harness.trajectory import CALLS, Trajectory

log = logging.getLogger("harness")

LLM_WORKERS = config.LLM_CONCURRENCY
JSON_TYPES = ("math", "data_analysis", "search")

# resolved lazily (importing them imports openjiuwen); tests may pre-set them
solve_group = None
solve_batch = None
json_route_batchable = None
json_route_batch_max = None
solve_with_program = None
fix_coding = None
arbitrate = None
BudgetExhausted = None


# infeasible notes that do not depend on the model's reading of the question
_MODEL_INDEPENDENT = ("division by zero", "start entity not found")


# name and model calls of the phase-2 unit running in this context (None outside phase 2)
_UNIT: ContextVar[dict | None] = ContextVar("harness_unit", default=None)


def _take_calls() -> dict:
    """Trajectory fields for the calls made by the current unit since the last take."""
    unit = _UNIT.get()
    if unit is None:
        return {"cost": 0}
    calls = list(unit["calls"])
    unit["calls"].clear()
    return {"unit": unit["name"], "model": calls[-1]["model"] if calls else None,
            "cost": sum(c["cost"] for c in calls), "calls": calls}


def _final_2a(res: SolveResult) -> bool:
    """2a settles `solved`, and `infeasible` only for model-independent verdicts; the rest goes to 2b."""
    if res.status == "solved":
        return True
    return res.status == "infeasible" and any(m in (res.note or "") for m in _MODEL_INDEPENDENT)


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
        if mod is None:
            res = SolveResult("unparsed", note="unknown task type")
        elif task.task_type == "coding":  # the solver's own limit stays inside the parent's kill timeout
            res = mod.solve(task, time_limit=config.coding_time_limit())
        else:
            res = mod.solve(task)
    except BaseException as e:  # the child must always answer
        res = SolveResult("unparsed", note=f"solver crashed: {e!r}")
    out_queue.put((task.task_id, res))


def _kill(proc) -> None:
    from harness.proc import kill_tree_of_process
    kill_tree_of_process(proc)
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
    killed: set = set()  # tids whose child was killed: a result arriving later is not trusted

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
            except queue_mod.Empty:
                return
            except Exception:
                log.exception("phase 1: bad queue message")
                return
            if tid in killed:
                timeout = 0.0
                continue
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
                limit = config.phase1_timeout(task.task_type)
                if not proc.is_alive():
                    drain(0.3)  # the result may still be in the pipe
                    if tid in running:
                        running.pop(tid)
                        finish(task, SolveResult("unparsed", note="solver process died"), start)
                elif time.monotonic() - start > limit or over:
                    killed.add(tid)
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
    global solve_group, solve_batch, json_route_batchable, json_route_batch_max, solve_with_program, fix_coding, arbitrate, BudgetExhausted
    from harness import arbitrate as arbitrate_mod, json_route, llm, llm_routes
    if solve_group is None:
        solve_group = json_route.solve_group
    if solve_batch is None:
        solve_batch = json_route.solve_batch
    if json_route_batchable is None:
        json_route_batchable = json_route.batchable
    if json_route_batch_max is None:
        json_route_batch_max = lambda: json_route.BATCH_MAX  # noqa: E731
    if solve_with_program is None:
        solve_with_program = llm_routes.solve_with_program
    if fix_coding is None:
        fix_coding = llm_routes.fix_coding
    if arbitrate is None:
        arbitrate = arbitrate_mod.arbitrate
    BudgetExhausted = llm.BudgetExhausted


async def _drive(units: list, t0: float, deadline_s: float, ledger: TokenLedger,
                 state: dict) -> None:
    """Run ordered units on LLM_WORKERS workers; stop issuing on exhaustion or near the deadline."""
    it = iter(units)

    async def worker() -> None:
        while not state["stop"]:
            if time.monotonic() - t0 > deadline_s - config.stop_new_before_deadline_s():
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
            remaining = deadline_s - (time.monotonic() - t0) - 5
            calls: list = []
            tokens = (_UNIT.set({"name": unit["name"], "calls": calls}), CALLS.set(calls))
            try:
                await asyncio.wait_for(unit["run"](), remaining)
            except asyncio.TimeoutError:
                state["traj"].record({"phase": 2, "task_id": unit["name"], "status": "deadline",
                                      **_take_calls(), "elapsed": round(time.monotonic() - start, 3)})
            except BudgetExhausted:
                state["stop"] = True
                ledger.mark_exhausted()
                state["traj"].record({"phase": 2, "task_id": unit["name"], "status": "budget_exhausted",
                                      **_take_calls(), "elapsed": round(time.monotonic() - start, 3)})
            except Exception as e:
                log.exception("unit %s failed", unit["name"])
                state["traj"].record({"phase": 2, "task_id": unit["name"], "status": "error",
                                      **_take_calls(), "note": repr(e)[:200],
                                      "elapsed": round(time.monotonic() - start, 3)})
            finally:
                if calls:  # no settled task took them (e.g. a 2a group with no final answer)
                    state["traj"].record({"phase": 2, "task_id": unit["name"], "status": "unit",
                                          **_take_calls(), "elapsed": round(time.monotonic() - start, 3)})
                _UNIT.reset(tokens[0])
                CALLS.reset(tokens[1])

    await asyncio.gather(*(worker() for _ in range(max(1, LLM_WORKERS))))


def run(tasks_dir: Path, budget: float, client, deadline_s: float | None = None,
        phase1_workers: int = config.PHASE1_WORKERS) -> dict:
    if deadline_s is None:
        deadline_s = config.deadline_s()
    t0 = time.monotonic()
    traj = Trajectory(_log_dir())
    ledger = getattr(client, "ledger", None) or TokenLedger(budget, budget * config.RESERVE_FRACTION)
    status: dict = {}
    spent0 = ledger.spent
    try:
        try:
            tasks = load_tasks(Path(tasks_dir))
        except Exception:
            log.exception("cannot load tasks from %s", tasks_dir)
            tasks = []
        by_id = {t.task_id: t for t in tasks}
        tentative: dict[str, SolveResult] = {}   # written but not final: phase 2 may improve it
        guesses: dict[str, SolveResult] = {}     # unanswered tasks with a stored best guess

        def settle(phase: int, task: Task, res: SolveResult, elapsed: float) -> None:
            st = res.status
            prior_tentative = task.task_id in tentative
            if st == "solved" and res.answer is None:
                st = "unparsed"
            elif st == "solved":
                try:
                    write_answer(task, res.answer)
                except Exception as e:
                    log.exception("cannot write answer for %s", task.task_id)
                    st, res = "unparsed", SolveResult("unparsed", note=f"write failed: {e!r}")
            if st == "solved":
                guesses.pop(task.task_id, None)
                if res.tentative:
                    tentative[task.task_id] = res
                else:
                    tentative.pop(task.task_id, None)
            elif prior_tentative and phase >= 2:
                st = "solved"   # a failed later attempt never takes back a written answer
            elif res.status == "infeasible" and phase >= 2:
                guesses.pop(task.task_id, None)   # a phase-2 verdict withdraws the stored guess
            elif (res.extra or {}).get("best_guess") is not None:
                # a value that failed the shape check only fills an empty slot: it never replaces a stored guess
                if task.task_id not in guesses or not (res.extra or {}).get("shape_mismatch"):
                    guesses[task.task_id] = res
            status[task.task_id] = st
            # the unit's own model calls (a group's go to its first settled member), not a ledger delta
            traj.record({"phase": phase, "task_id": task.task_id, "status": st, **_take_calls(),
                         "elapsed": round(elapsed, 3), "note": (res.note or "")[:200]})

        _phase1(tasks, phase1_workers, t0, deadline_s, lambda t, r, e: settle(1, t, r, e))

        left = [by_id[i] for i, s in status.items() if s == "unparsed" or i in tentative]
        if client is not None and left:
            _phase2(left, client, ledger, t0, deadline_s, settle, status, traj, tentative)
        _write_guesses(by_id, guesses, tentative, status)
    except Exception:
        log.exception("scheduler failed")
    finally:
        try:
            # charges outside any recorded call keep the trajectory total equal to the ledger
            extra = ledger.spent - spent0 - traj.cost_total
            if abs(extra) > 1e-6:
                traj.record({"phase": 2, "task_id": None, "status": "unattributed", "cost": extra})
        except Exception:
            log.exception("cost reconciliation failed")
        traj.close()
    counts = {"solved": 0, "infeasible": 0, "unparsed": 0}
    for s in status.values():
        counts[s if s in counts else "unparsed"] += 1
    counts["spent"] = ledger.spent
    return counts


def _write_guesses(by_id: dict, guesses: dict, tentative: dict, status: dict) -> None:
    """Last pass: a stored best guess becomes the answer of a task that has none yet.
    Never touches a task that already has an answer file (final or tentative)."""
    for tid, res in guesses.items():
        task = by_id.get(tid)
        try:
            if task is None or (task.dir / "answer.txt").exists():
                continue
            write_answer(task, str(res.extra["best_guess"]))
            status[tid] = "solved"
        except Exception:
            log.exception("cannot write best guess for %s", tid)


def _phase2(*args) -> None:
    # one event loop for the whole phase: openjiuwen caches an HTTP pool bound to the first loop
    try:
        asyncio.run(_phase2_async(*args))
    except Exception:
        log.exception("phase 2 failed")


async def _phase2_async(left, client, ledger, t0, deadline_s, settle, status, traj,
                        tentative=None) -> None:
    tentative = tentative if tentative is not None else {}
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
    def settle_group(members, results, el):
        for task, _ in members:
            res = results.get(task.task_id)
            # a non-final result with a best guess is settled too: it stays unparsed (2b still runs)
            # and its guess is written by the last pass if nothing better comes
            if res is not None and (_final_2a(res) or (res.extra or {}).get("best_guess") is not None):
                settle(2, task, res, el)

    def make_unit(call, members, est, name):
        """`call()` returns {task_id: result}; its model calls and cost go to the first settled member."""
        async def go():
            start = time.monotonic()
            exhausted = None
            try:
                results = await call()
            except BudgetExhausted as e:  # may carry the answers computed before it (shape retry)
                results, exhausted = getattr(e, "results", None) or {}, e
            settle_group(members, results, time.monotonic() - start)
            if exhausted is not None:
                raise exhausted
        reward = sum(t.reward for t, _ in members)
        return {"prio": reward / est, "estimate": est, "run": go, "name": name}

    singles, small = [], {}
    for (ttype, template), members in groups.items():
        if config.batch_2a() and json_route_batchable(ttype, template, members):
            small.setdefault(ttype, []).append((template, members))
        else:
            singles.append((ttype, template, members))
    for ttype, items in small.items():
        for i in range(0, len(items), json_route_batch_max()):
            chunk = items[i:i + json_route_batch_max()]
            if len(chunk) == 1:
                singles.append((ttype, chunk[0][0], chunk[0][1]))
                continue
            allm = [m for _, ms in chunk for m in ms]
            est = sum(estimate_json_tokens(t) for t, _ in chunk)  # the shared system prompt makes it an upper bound
            units.append(make_unit(lambda ttype=ttype, chunk=chunk: solve_batch(ttype, chunk, client), allm, est,
                                   f"batch:{ttype}:{len(chunk)}"))
    for ttype, template, members in singles:
        # expected cost of one spec call; the rare shape retry on MODEL_ALT (up to 2 more) is left to the reserve
        est = estimate_json_tokens(template)
        units.append(make_unit(lambda ttype=ttype, template=template, members=members:
                               solve_group(ttype, template, members, client),
                               members, est, f"group:{ttype}:{len(members)}"))
    units.sort(key=lambda u: -u["prio"])
    await _drive(units, t0, deadline_s, ledger, state)

    # 2c: model arbitration among behaviourally different green fixes (cheap: before 2b)
    units = []
    for task in [t for t in left if (tentative.get(t.task_id) is not None
                                     and (tentative[t.task_id].extra or {}).get("ambiguous"))]:
        extra = tentative[task.task_id].extra
        # expected cost: one call per arbitration model. The rare retry without the regex (a second call per
        # model, only when a reply cannot be read) is not counted; the ledger reserve covers it
        est = (len(str(extra.get("ambiguous"))) // 30 + 10) * len(config.ARBITRATE_MODELS)

        def make(task=task, extra=extra):
            async def go():
                start = time.monotonic()
                res = await arbitrate(task, extra, client)
                if res.status == "solved":
                    res = SolveResult("solved", res.answer, res.note)  # final: no longer tentative
                settle(2, task, res, time.monotonic() - start)
            return go
        units.append({"prio": 1e9, "estimate": est, "run": make(), "name": f"arbitrate:{task.task_id}"})
    await _drive(units, t0, deadline_s, ledger, state)

    # 2b: per-task routes for whatever is still unparsed or only tentatively answered
    # (arbitrated tasks are final; tentative ones with candidates wait for 2c above)
    units = []
    for task in [t for t in left if status.get(t.task_id) == "unparsed"
                 or (t.task_id in tentative and not (tentative[t.task_id].extra or {}).get("ambiguous"))]:
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
    await _drive(units, t0, deadline_s, ledger, state)

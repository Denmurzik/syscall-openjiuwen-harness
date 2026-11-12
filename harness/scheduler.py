"""Two-phase orchestration: free deterministic solvers first, then budgeted model routes."""
import asyncio
import logging
import multiprocessing
import os
import queue as queue_mod
import statistics
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
solve_with_reason = None
fix_coding = None
arbitrate = None
verify_fix = None
confirm_program = None   # the program route of a confirmation vote (task 40); tests may pre-set it
prose_agreement = None   # the second and third attempts on a prose answer (task 79); tests may pre-set it
is_prose_task = None
second_opinion = None
BudgetExhausted = None

REASON_PASS_MAX_CALLS = 10      # hard cap of the final reasoning pass, whatever the ledger shows
REASON_PASS_SPEND_LIMIT = 0.5   # no final reasoning pass once this share of the budget is spent
REASON_CALL_TOKENS = 3000       # output limit of solve_with_reason's one call: what the ledger must hold for it


def _reason_fits(ledger: TokenLedger, task: Task) -> bool:
    """A whole MODEL_REASON call fits the ledger. A call cut below its limit reasons to the end and answers nothing
    (the ledger stops it only under MIN_REASONING_CALL_TOKENS): such a task goes the cheap way instead."""
    return ledger.can_afford(estimate_tokens(task) + REASON_CALL_TOKENS)


REASON_FAST_MARGIN_S = 60       # a FAST program call and its run after a REASON call (usually 10-30 s; sandbox 20 s)


def _reason_2b(ledger: TokenLedger, task: Task, t0: float, deadline_s: float) -> bool:
    """A 2b math task asks MODEL_REASON first: HARNESS_REASON_MATH on, a whole call fits the ledger, less than
    REASON_PASS_SPEND_LIMIT of the budget is spent (the final pass's gate), and the time left before new units stop
    holds a REASON call to its timeout plus the FAST fallback (else the unit would be cancelled before FAST runs).
    Otherwise the FAST program alone, as in round 9."""
    if not config.reason_math() or ledger.spent >= REASON_PASS_SPEND_LIMIT * ledger.budget:
        return False
    left_s = deadline_s - config.stop_new_before_deadline_s() - (time.monotonic() - t0)
    return left_s >= config.llm_timeout() + REASON_FAST_MARGIN_S and _reason_fits(ledger, task)


def _prose_more(ledger: TokenLedger, task: Task, t0: float, deadline_s: float, reason: bool) -> bool:
    """One more attempt on a prose answer (task 79) may start: less than REASON_PASS_SPEND_LIMIT of the budget is
    spent, new units are not stopped yet, and the ledger holds the attempt (a REASON call whole, see _reason_fits).
    A REASON attempt also needs the time of its call to its timeout before new units stop: one cut by the deadline
    is paid and answers nothing."""
    if ledger.spent >= REASON_PASS_SPEND_LIMIT * ledger.budget:
        return False
    left_s = deadline_s - config.stop_new_before_deadline_s() - (time.monotonic() - t0)
    if left_s < 0 or (reason and left_s < config.llm_timeout()):
        return False
    return _reason_fits(ledger, task) if reason else ledger.can_afford(estimate_tokens(task))


def _reason_verdict(res: SolveResult) -> bool:
    """MODEL_REASON's own output decided `res` (an answer, IMPOSSIBLE, or a reply that gave nothing usable), not the
    infrastructure (model unavailable, paused or timed out; call cut by the budget)."""
    note = res.note or ""
    return not note.startswith(_LLM_UNAVAILABLE) and note != "reason program cut by the budget limit"


def _reason_votes(task: Task) -> bool:
    """The 2a answer of `task` is confirmed by MODEL_REASON (`confirm_reason`), not by two programs. Off by default
    (HARNESS_CONFIRM_MATH); HARNESS_REASON_MATH=0 turns it off too (round 9)."""
    return (task.task_type not in config.CONFIRM_TYPES and task.task_type in config.CONFIRM_REASON_TYPES
            and config.reason_math() and config.confirm_math())


SECOND_TYPES = ("search", "data_analysis")   # 2b types whose first IMPOSSIBLE gets a MODEL_ALT opinion (task 86)


# infeasible notes that do not depend on the model's reading of the question
_MODEL_INDEPENDENT = ("division by zero", "start entity not found")
_LLM_UNAVAILABLE = "llm unavailable"  # note of a result whose model call never got an answer


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


def _phase1_target(out_queue, task: Task, solver_name=None, known_texts=None) -> None:
    """Child process body: run one deterministic solver, send (task_id, SolveResult). `known_texts`: function
    texts mined from the run's coding tasks (`_mine_texts`: {"texts": {name: text}, "final": [names]}), added to
    the coding solver's known texts."""
    try:
        mod = _solver_for(task.task_type)
        if mod is None:
            res = SolveResult("unparsed", note="unknown task type")
        elif task.task_type == "coding":  # the solver's own limit stays inside the parent's kill timeout
            if known_texts:
                mod.install_known_texts(known_texts.get("texts"), known_texts.get("final", ()))
            res = mod.solve(task, time_limit=config.coding_time_limit())
        else:
            res = mod.solve(task)
    except BaseException as e:  # the child must always answer
        res = SolveResult("unparsed", note=f"solver crashed: {e!r}")
    out_queue.put((task.task_id, res))


MINE_MIN_TASKS = 3            # fewer coding tasks than this cannot give two votes worth trusting
MINE_DEADLINE_SHARE = 0.1     # mining takes at most this share of the time left (and MINE_BUDGET_S)


def _mine_texts(tasks: list, t0: float | None = None, deadline_s: float | None = None) -> dict:
    """Function texts that several coding tasks of this run hold with green tests (`coding_solver.mine_known_texts`),
    for the phase-1 coding children: {"texts": {name: text}, "final": [names]}, or {} when there are none. Off
    (HARNESS_MINE_TEXTS=0) or fewer than MINE_MIN_TASKS coding tasks: nothing runs. Never fails the run: without
    them phase 1 is as in round 10."""
    if not config.mine_texts():
        return {}
    try:
        coding = [t.workspace for t in tasks if t.task_type == "coding" and t.workspace is not None]
        if len(coding) < MINE_MIN_TASKS:
            return {}
        from harness.solvers import coding_solver
        budget_s = coding_solver.MINE_BUDGET_S
        if t0 is not None and deadline_s is not None:
            budget_s = min(budget_s, MINE_DEADLINE_SHARE * max(0.0, deadline_s - (time.monotonic() - t0)))
        start = time.monotonic()
        texts, final = coding_solver.mine_known_texts(coding, budget_s=budget_s)
        log.info("mined texts of %d functions from %d coding tasks in %.1f s: %s; final: %s", len(texts),
                 len(coding), time.monotonic() - start, sorted(texts), sorted(final))
        return {"texts": texts, "final": sorted(final)} if texts else {}
    except Exception:
        log.exception("mining function texts failed; phase 1 goes on without them")
        return {}


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


def _phase1(tasks: list, workers: int, t0: float, deadline_s: float, on_result, known_texts=None) -> None:
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
                    args = (out_queue, task, None)
                    if known_texts and task.task_type == "coding":
                        args += (known_texts,)
                    proc = ctx.Process(target=_phase1_target, args=args, daemon=True)
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
    global solve_group, solve_batch, json_route_batchable, json_route_batch_max, solve_with_program, solve_with_reason, fix_coding, arbitrate, verify_fix, confirm_program, prose_agreement, is_prose_task, second_opinion, BudgetExhausted
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
    if solve_with_reason is None:
        solve_with_reason = llm_routes.solve_with_reason
    if fix_coding is None:
        fix_coding = llm_routes.fix_coding
    if arbitrate is None:
        arbitrate = arbitrate_mod.arbitrate
    if verify_fix is None:
        verify_fix = arbitrate_mod.verify_fix
    if confirm_program is None:
        confirm_program = llm_routes.solve_with_program
    if prose_agreement is None:
        prose_agreement = llm_routes.prose_agreement
    if is_prose_task is None:
        is_prose_task = llm_routes._is_prose_task
    if second_opinion is None:
        second_opinion = llm_routes.second_opinion
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
        soft_infeasible: set[str] = set()        # a model IMPOSSIBLE: not proven, the final REASON pass may still try
        proven_infeasible: set[str] = set()      # a model-independent verdict in phase 2: a guess on disk stays there
        written: set[str] = set()                # tasks whose answer.txt holds a guess of ours, not a final answer

        def put_guess(task: Task, res: SolveResult) -> None:
            """The guess goes to disk at once: the run may be stopped silently at the end of the budget.
            Only an empty slot or an earlier guess of ours is written, never a final answer."""
            tid = task.task_id
            try:
                if (task.dir / "answer.txt").exists() and tid not in written:
                    return
                write_answer(task, str(res.extra["best_guess"]))
                written.add(tid)
            except Exception:
                log.exception("cannot write best guess for %s", tid)

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
                written.discard(task.task_id)   # the final answer replaced our guess
                if res.tentative:
                    tentative[task.task_id] = res
                else:
                    tentative.pop(task.task_id, None)
            elif prior_tentative and phase >= 2:
                st = "solved"   # a failed later attempt never takes back a written answer
            elif res.status == "infeasible" and phase >= 2 and not any(m in (res.note or "") for m in _MODEL_INDEPENDENT):
                soft_infeasible.add(task.task_id)
                pass   # the model's IMPOSSIBLE is often wrong and a wrong answer costs nothing: the guess stays
            elif res.status == "infeasible" and phase >= 2:
                # a model-independent verdict: no more model calls on the task, but the stored guess and its
                # answer.txt stay. An answer to an infeasible task is not counted; if the verdict is wrong, it may score
                proven_infeasible.add(task.task_id)
                soft_infeasible.discard(task.task_id)
            elif (res.extra or {}).get("best_guess") is not None:
                # a value that failed the shape check only fills an empty slot: it never replaces a stored guess
                if task.task_id not in guesses or not (res.extra or {}).get("shape_mismatch"):
                    guesses[task.task_id] = res
                    put_guess(task, res)
            status[task.task_id] = st
            # the unit's own model calls (a group's go to its first settled member), not a ledger delta
            traj.record({"phase": phase, "task_id": task.task_id, "status": st, **_take_calls(),
                         "elapsed": round(elapsed, 3), "note": (res.note or "")[:200]})

        mined = _mine_texts(tasks, t0, deadline_s)
        _phase1(tasks, phase1_workers, t0, deadline_s, lambda t, r, e: settle(1, t, r, e),
                **({"known_texts": mined} if mined else {}))

        left = sorted((by_id[i] for i, s in status.items() if s == "unparsed" or i in tentative),
                      key=lambda t: t.task_id)  # batch composition depends only on the task set
        # the audit of rival green forms (HARNESS_AUDIT_FORMS) runs at the very end of phase 2, even when phase 1
        # left nothing open. Off, or too few coding tasks: phase 2 is called exactly as in round 13
        audit_tasks = _audit_tasks(tasks) if client is not None else []
        review_tasks = _review_tasks(tasks) if client is not None else []
        if client is not None and (left or audit_tasks or review_tasks):
            _phase2(left, client, ledger, t0, deadline_s, settle, status, traj, tentative,
                    [t.reward for t in tasks], soft_infeasible=soft_infeasible,
                    **({"audit_tasks": audit_tasks} if audit_tasks else {}),
                    **({"review_tasks": review_tasks, "review_known": mined} if review_tasks else {}))
        _write_guesses(by_id, guesses, tentative, status, written, proven_infeasible)
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


def _write_guesses(by_id: dict, guesses: dict, tentative: dict, status: dict, written=(), proven=()) -> None:
    """Last pass: a stored best guess becomes the answer of a task that has none yet (most are on disk
    already: they only count as solved). Never touches a task that already has an answer file. A task in `proven`
    (a model-independent infeasible verdict) keeps its guess on disk but is still counted infeasible."""
    for tid, res in guesses.items():
        task = by_id.get(tid)
        settled = "infeasible" if tid in proven and status.get(tid) == "infeasible" else "solved"
        try:
            if tid in written:
                status[tid] = settled
                continue
            if task is None or (task.dir / "answer.txt").exists():
                continue
            write_answer(task, str(res.extra["best_guess"]))
            status[tid] = settled
        except Exception:
            log.exception("cannot write best guess for %s", tid)


def _phase2(*args, **kw) -> None:
    # one event loop for the whole phase: openjiuwen caches an HTTP pool bound to the first loop
    try:
        asyncio.run(_phase2_async(*args, **kw))
    except Exception:
        log.exception("phase 2 failed")


async def _phase2_async(left, client, ledger, t0, deadline_s, settle, status, traj,
                        tentative=None, rewards=None, soft_infeasible=(), audit_tasks=(), review_tasks=(),
                        review_known=None) -> None:
    tentative = tentative if tentative is not None else {}
    try:
        os.chdir(_log_dir())  # openjiuwen writes ./logs relative to cwd
    except OSError:
        pass
    state = {"stop": False, "traj": traj}
    if not left:   # phase 1 settled everything: only the audit of rival green forms may have work
        await _audit_stage(audit_tasks, client, ledger, t0, deadline_s, status, traj, state)
        await _review_stage(review_tasks, client, ledger, t0, deadline_s, status, traj, state, review_known)
        return
    _load_phase2()
    from harness.templates import group

    # 2a: one JSON-spec call per template group
    units = []
    try:
        groups = group([t for t in left if t.task_type in JSON_TYPES])
    except Exception:
        log.exception("grouping failed")
        groups = {}
    group_of = {t.task_id: key for key, members in groups.items() for t, _ in members}
    answered_2a: dict = {}   # task_id -> (task, answer written by 2a): the confirmation stage's candidates

    def settle_group(members, results, el):
        for task, _ in members:
            res = results.get(task.task_id)
            # a non-final result with a best guess is settled too: it stays unparsed (2b still runs)
            # and its guess is written by the last pass if nothing better comes
            if res is not None and (_final_2a(res) or (res.extra or {}).get("best_guess") is not None):
                settle(2, task, res, el)
                if (res.status == "solved" and status.get(task.task_id) == "solved"
                        and task.task_id not in tentative):
                    answered_2a[task.task_id] = (task, str(res.answer))

    def make_unit(call, members, est, name, stream=False):
        """`call()` returns {task_id: result}; its model calls and cost go to the first settled member.
        With `stream` the call settles each group itself as soon as it is ready (call(on_group)), so a
        cancelled unit keeps the groups that were done."""
        async def go():
            start = time.monotonic()
            exhausted = None
            if stream:
                await call(lambda members, results: settle_group(members, results, time.monotonic() - start))
                return
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
            units.append(make_unit(lambda on_group, ttype=ttype, chunk=chunk: solve_batch(ttype, chunk, client, on_group=on_group),
                                   allm, est, f"batch:{ttype}:{len(chunk)}", stream=True))
    for ttype, template, members in singles:
        # expected cost of one spec call; the rare shape retry on MODEL_ALT (up to 2 more) is left to the reserve
        est = estimate_json_tokens(template)
        units.append(make_unit(lambda ttype=ttype, template=template, members=members:
                               solve_group(ttype, template, members, client),
                               members, est, f"group:{ttype}:{len(members)}"))
    units.sort(key=lambda u: -u["prio"])
    await _drive(units, t0, deadline_s, ledger, state)

    unavailable: list = []  # factories of the 2c/2b units whose model call got no answer
    reasoned: set = set()   # math tasks MODEL_REASON was asked in 2b: the final REASON pass does not ask again

    # 2c: model arbitration among behaviourally different green fixes (cheap: before 2b)
    def unit_2c(task):
        extra = tentative[task.task_id].extra
        # expected cost: one call per arbitration model. The rare retry without the regex (a second call per
        # model, only when a reply cannot be read) is not counted; the ledger reserve covers it
        est = (len(str(extra.get("ambiguous"))) // 30 + 10) * len(config.ARBITRATE_MODELS)

        async def go():
            start = time.monotonic()
            res = await arbitrate(task, extra, client)
            if res.status == "solved":
                res = SolveResult("solved", res.answer, res.note)  # final: no longer tentative
            settle(2, task, res, time.monotonic() - start)
            if (res.note or "").startswith(_LLM_UNAVAILABLE):
                unavailable.append(lambda: unit_2c(task))
        return {"prio": 1e9, "estimate": est, "run": go, "name": f"arbitrate:{task.task_id}"}

    units = [unit_2c(t) for t in left if (tentative.get(t.task_id) is not None
                                          and (tentative[t.task_id].extra or {}).get("ambiguous"))]
    await _drive(units, t0, deadline_s, ledger, state)

    def prose_on(task) -> bool:
        """The task takes the prose agreement (task 79): HARNESS_PROSE_AGREE on and a prose knowledge base."""
        try:
            return task.task_type == "search" and config.prose_agree() and is_prose_task(task)
        except Exception:
            log.exception("prose check on %s failed", task.task_id)
            return False

    async def agree_prose(task, start):
        """A prose answer needs a second attempt that agrees (task 79). FAST's answer goes to disk before the extra
        attempts: a deadline or a silent stop of the run in them keeps it. A FAST route that raised is a failed
        first attempt: ALT and REASON still try."""
        try:
            first = await solve_with_program(task, client)
        except BudgetExhausted:
            raise
        except Exception as e:
            log.warning("program route on %s failed: %r", task.task_id, e)
            first = SolveResult("unparsed", note=f"program route failed: {e!r}"[:200])
        more = lambda reason: _prose_more(ledger, task, t0, deadline_s, reason)  # noqa: E731
        try:
            if first.status == "solved" and more(False):
                settle(2, task, first, time.monotonic() - start)
            res = await prose_agreement(task, client, first, more,
                                        keep=lambda best: settle(2, task, best, time.monotonic() - start))
        except BudgetExhausted:
            raise
        except Exception:   # the extra attempts are lost, FAST's answer is not
            log.exception("prose agreement on %s failed", task.task_id)
            return first
        if (res.extra or {}).get("reasoned"):
            reasoned.add(task.task_id)   # REASON gave its verdict: the final pass does not ask again
        return res

    def second_due(task, res) -> bool:
        """HARNESS_SECOND_IMPOSSIBLE (task 86): FAST's program on a search or data task that is not a prose one found
        it infeasible by its own reading (`first_impossible`), and the gates of another attempt hold (less than half
        of the budget spent, new units not stopped, the ledger holds the call)."""
        try:
            from harness.llm_routes import first_impossible
            return (config.second_impossible() and task.task_type in SECOND_TYPES
                    and first_impossible(res) and not is_prose_task(task)
                    and _prose_more(ledger, task, t0, deadline_s, False))
        except Exception:
            log.exception("second-opinion check on %s failed", task.task_id)
            return False

    # 2b: per-task routes for whatever is still unparsed or only tentatively answered
    # (arbitrated tasks are final; tentative ones with candidates wait for 2c above)
    def unit_2b(task, resume=None):
        est = estimate_tokens(task)

        async def go():
            start = time.monotonic()
            if task.task_type == "coding":
                # a partial repair of phase 1 carries the file before its mutations: the ladder asks for the whole
                # repair from it first (task 63). Without one the call is the round-10 call
                restart = ((tentative.get(task.task_id) and tentative[task.task_id].extra) or {}).get("restart")
                args = (task, client, resume) if resume else (task, client)
                res = await (fix_coding(*args, restart=restart) if restart else fix_coding(*args))
            elif (task.task_type == "math" and task.task_id not in reasoned
                  and _reason_2b(ledger, task, t0, deadline_s)):
                # 2a declined it (not "how much is left at the end") or failed: such a question takes reasoning,
                # not a transcription. The FAST program is the fallback, and a guess for a task still open.
                # A task in `reasoned` (REASON gave its verdict, then FAST found the model unavailable) retries FAST
                try:
                    res = await solve_with_reason(task, client)
                    if _reason_verdict(res):
                        reasoned.add(task.task_id)
                except BudgetExhausted:
                    raise
                except Exception as e:  # the route is lost, the task is not (and the final pass may still ask)
                    log.warning("reason route on %s failed: %r", task.task_id, e)
                    res = SolveResult("unparsed", note=f"reason route failed: {e!r}"[:200])
                # IMPOSSIBLE is trusted (REASON 20 of 20 on feasible math; on v3 math_9319 the FAST fallback wrote an
                # answer for an infeasible task): settled below, no FAST program
                if res.status not in ("solved", "infeasible"):
                    if (res.extra or {}).get("best_guess") is not None:
                        settle(2, task, res, time.monotonic() - start)
                    res = await solve_with_program(task, client)
            elif prose_on(task):
                res = await agree_prose(task, start)
            else:
                res = await solve_with_program(task, client)
                if second_due(task, res):
                    try:
                        res = await second_opinion(task, client, res)
                    except BudgetExhausted:
                        raise
                    except Exception:   # the second opinion is lost, FAST's verdict is not
                        log.exception("second opinion on %s failed", task.task_id)
            settle(2, task, res, time.monotonic() - start)
            if (res.note or "").startswith(_LLM_UNAVAILABLE):
                ladder = (res.extra or {}).get("ladder")
                unavailable.append(lambda: unit_2b(task, ladder))
        return {"prio": task.reward / est, "estimate": est, "run": go, "name": task.task_id}

    units = [unit_2b(t) for t in left if status.get(t.task_id) == "unparsed"
             or (t.task_id in tentative and not (tentative[t.task_id].extra or {}).keys() & {"ambiguous", "verify"})]
    units.sort(key=lambda u: -u["prio"])
    await _drive(units, t0, deadline_s, ledger, state)

    # one more pass over what found the model unavailable, unless the run is over (dead proxy, spent budget,
    # deadline). A proxy that is still failing here holds these calls until it recovers or is declared dead
    if unavailable and not state["stop"] and not ledger.exhausted:
        units = [make() for make in unavailable]
        units.sort(key=lambda u: -u["prio"])
        await _drive(units, t0, deadline_s, ledger, state)

    # final pass: the cheap routes failed these. One MODEL_REASON program each for the best-paid few, while less
    # than half of the budget is spent. The call cap does not depend on the ledger (a proxy may report no usage)
    if not state["stop"] and not ledger.exhausted and ledger.spent < REASON_PASS_SPEND_LIMIT * ledger.budget:
        rewards = rewards if rewards is not None else [t.reward for t in left]
        median = statistics.median(rewards) if rewards else 0
        cands = [t for t in left if (status.get(t.task_id) == "unparsed" or t.task_id in soft_infeasible) and t.task_type in JSON_TYPES
                 and t.task_id not in tentative and t.reward >= median and t.task_id not in reasoned]
        cands.sort(key=lambda t: -t.reward)

        def unit_reason(task):
            async def go():
                if ledger.spent >= REASON_PASS_SPEND_LIMIT * ledger.budget:
                    return
                start = time.monotonic()
                res = await solve_with_reason(task, client)
                settle(2, task, res, time.monotonic() - start)
            return {"prio": task.reward, "estimate": 3000, "run": go, "name": f"reason:{task.task_id}"}

        await _drive([unit_reason(t) for t in cands[:REASON_PASS_MAX_CALLS]], t0, deadline_s, ledger, state)

    # 2d: an independent model check of green single coding fixes. Last of all, after every pass over open tasks:
    # an open task is worth more than a second opinion on an answered one, and the written fix stays whatever
    # happens here (a budget or deadline stop in 2d cancels nothing that ran before it)
    unchecked: list = []  # 2d units whose model call got no answer: one pass of their own

    def unit_2d(task, extra):
        est = len(str(extra)) // 30 + config.VERIFY_MAX_TOKENS  # the arbitration behind a difference: left to the reserve

        async def go():
            start = time.monotonic()
            res = await verify_fix(task, extra, client)
            if res.status == "solved":
                res = SolveResult("solved", res.answer, res.note)  # final: no longer tentative
            settle(2, task, res, time.monotonic() - start)
            if (res.note or "").startswith(_LLM_UNAVAILABLE):
                resume = res.extra or extra
                unchecked.append(lambda: unit_2d(task, resume))
        return {"prio": task.reward / est, "estimate": est, "run": go, "name": f"verify:{task.task_id}"}

    units_2d = [unit_2d(t, tentative[t.task_id].extra) for t in left
                if t.task_id in tentative and (tentative[t.task_id].extra or {}).get("verify")]
    units_2d.sort(key=lambda u: -u["prio"])

    # confirmation of 2a answers by independent programs, in reward order. After 2b, the retry pass and the final
    # REASON pass: an open task is worth more than a second opinion, and the REASON pass's spend gate must not see
    # the votes. Before 2d: per unit it is worth about ten times more (a wrong 2a answer is 4 in 41 on the expert's
    # sets, at ~150 a math or data task; a wrong green fix about 1 in 20, at ~450 a check), which decides the order if the real
    # budget ends silently. Within the ledger and the deadline 2d keeps what it needs: the votes leave its estimate
    # and its time untouched
    if config.confirm_2a() and answered_2a and not state["stop"] and not ledger.exhausted:
        await _drive(_confirm_units(answered_2a, group_of, client, ledger, t0, deadline_s, settle, status, traj,
                                    _reserve_2d(units_2d), len(units_2d)), t0, deadline_s, ledger, state)

    await _drive(units_2d, t0, deadline_s, ledger, state)

    if unchecked and not state["stop"] and not ledger.exhausted:
        units = [make() for make in unchecked]
        units.sort(key=lambda u: -u["prio"])
        await _drive(units, t0, deadline_s, ledger, state)

    # last of all: every answer of the run is on disk, the audit only rewrites mylib.py of answered coding tasks
    await _audit_stage(audit_tasks, client, ledger, t0, deadline_s, status, traj, state)
    await _review_stage(review_tasks, client, ledger, t0, deadline_s, status, traj, state, review_known)


AUDIT_MIN_TASKS = 3        # fewer coding tasks cannot show one function in rival forms worth a question
AUDIT_SPEND_LIMIT = 0.5    # no audit unit and no third opinion once this share of the budget is spent
AUDIT_MIN_LEFT_S = 60      # the stage starts only with this much time left before new units stop
AUDIT_CENSUS_SHARE = 0.1   # the census takes at most this share of that time (and audit.CENSUS_BUDGET_S)
AUDIT_MAX_FUNCS = 20       # functions asked about, the most-held first
AUDIT_MAX_REASON = 6       # third opinions (MODEL_REASON) on minority picks in a run


def _audit_tasks(tasks: list) -> list:
    """The coding tasks the audit stage looks at; [] when HARNESS_AUDIT_FORMS is off or they are too few."""
    if not config.audit_forms():
        return []
    coding = [t for t in tasks if t.task_type == "coding" and t.workspace is not None]
    return coding if len(coding) >= AUDIT_MIN_TASKS else []


async def _audit_stage(tasks, client, ledger, t0, deadline_s, status, traj, state) -> None:
    """HARNESS_AUDIT_FORMS: functions that stand green in rival forms across the answered coding tasks
    (`harness.audit`). One unit per function: two 8-token calls (MODEL_FAST, MODEL_ALT); the option both name replaces
    the other shown forms where every test file stays green, and an option the set holds less often needs MODEL_REASON.
    No rival forms: no call. The spend gate holds for every unit and every third opinion. Never fails the run."""
    if not tasks or state["stop"] or ledger.exhausted or ledger.spent >= AUDIT_SPEND_LIMIT * ledger.budget:
        return
    try:
        from harness import audit

        def left_s() -> float:
            return deadline_s - config.stop_new_before_deadline_s() - (time.monotonic() - t0)

        done = [t for t in tasks if status.get(t.task_id) == "solved" and (t.dir / "answer.txt").exists()]
        if left_s() < AUDIT_MIN_LEFT_S or len(done) < AUDIT_MIN_TASKS:
            return
        holders, rivals = await asyncio.to_thread(audit.census, done,
                                                  min(audit.CENSUS_BUDGET_S, AUDIT_CENSUS_SHARE * left_s()))
        if not rivals:
            return
        log.info("audit: rival forms of %s in %d green coding tasks", sorted(rivals), len(holders))
        _load_phase2()
        lock = asyncio.Lock()
        reasons = [0]

        def gate() -> bool:
            return not ledger.exhausted and ledger.spent < AUDIT_SPEND_LIMIT * ledger.budget

        def may_reason(est: float) -> bool:
            if reasons[0] >= AUDIT_MAX_REASON or not gate() or not ledger.can_afford(est):
                return False
            reasons[0] += 1
            return True

        def unit(name, forms):
            est = audit.unit_estimate(forms)

            async def go():
                if not gate():
                    return
                start = time.monotonic()
                winner, note = await audit.decide(client, name, forms, holders, may_reason)
                changed = []
                if winner is not None:
                    async with lock:
                        stop = t0 + deadline_s - config.stop_new_before_deadline_s()
                        changed = await asyncio.to_thread(audit.apply, name, winner, holders, stop)
                traj.record({"phase": 2, "task_id": f"audit:{name}", "status": "unit", **_take_calls(),
                             "elapsed": round(time.monotonic() - start, 3),
                             "note": f"{note}; {len(forms)} forms; replaced in {changed}"[:300]})
            return {"prio": sum(f["votes"] + f["untested"] for f in forms), "estimate": est, "run": go,
                    "name": f"audit:{name}"}

        units = [unit(n, fs) for n, fs in sorted(rivals.items())]
        units.sort(key=lambda u: -u["prio"])
        await _drive(units[:AUDIT_MAX_FUNCS], t0, deadline_s, ledger, state)
    except Exception:
        log.exception("audit stage failed; the files stay as they are")


REVIEW_SPEND_LIMIT = 0.5   # no review unit once this share of the budget is spent
REVIEW_MIN_LEFT_S = 60     # the stage starts only with this much time left before new units stop


def _review_tasks(tasks: list) -> list:
    """The coding tasks the review stage looks at; [] when HARNESS_REVIEW_CODING is off."""
    if not config.review_coding():
        return []
    return [t for t in tasks if t.task_type == "coding" and t.workspace is not None]


async def _review_stage(tasks, client, ledger, t0, deadline_s, status, traj, state, mined=None) -> None:
    """HARNESS_REVIEW_CODING: every answered coding file that holds a function outside the known texts is read by
    MODEL_FAST for bugs the visible tests do not catch; MODEL_ALT is asked only when FAST rewrote something, and a
    function is replaced only on their agreement with the visible tests green (`harness.review`). After everything
    else: the answers are on disk and a stop here loses nothing. `mined`: the texts mined from the set
    (`_mine_texts`), known like the examples' (never reviewed, never replaced). Never fails the run."""
    if not tasks or state["stop"] or ledger.exhausted or ledger.spent >= REVIEW_SPEND_LIMIT * ledger.budget:
        return
    try:
        from harness import review
        from harness.llm_routes import _target_file, _test_file

        if deadline_s - config.stop_new_before_deadline_s() - (time.monotonic() - t0) < REVIEW_MIN_LEFT_S:
            return
        known = review.known_texts(mined)
        # no probe or test run of the review after this: `_drive` cancels the unit, not its worker thread
        stop = t0 + deadline_s - max(config.stop_new_before_deadline_s(), 10.0)
        units = []
        for task in tasks:
            if status.get(task.task_id) != "solved" or not (task.dir / "answer.txt").exists():
                continue
            try:
                ws = Path(task.workspace)
                tests = (ws / _test_file(ws)).read_text(encoding="utf-8")
                src = (ws / _target_file(ws, tests)).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if not review.needs_review(src, known):
                continue

            def unit(task=task, est=review.estimate(src, tests)):
                async def go():
                    if ledger.exhausted or ledger.spent >= REVIEW_SPEND_LIMIT * ledger.budget:
                        return
                    start = time.monotonic()
                    changed, note = await review.review(task, client, known, stop)
                    traj.record({"phase": 2, "task_id": f"review:{task.task_id}", "status": "unit", **_take_calls(),
                                 "elapsed": round(time.monotonic() - start, 3), "note": note[:300]})
                return {"prio": task.reward / est, "estimate": est, "run": go, "name": f"review:{task.task_id}"}
            units.append(unit())
        if not units:
            return
        _load_phase2()
        units.sort(key=lambda u: -u["prio"])
        await _drive(units, t0, deadline_s, ledger, state)
    except Exception:
        log.exception("review stage failed; the files stay as they are")


def _reserve_2d(units: list) -> float:
    """What the confirmation stage leaves on the ledger for the 2d checks after it."""
    return sum(u["estimate"] for u in units)


def _confirm_time_left(t0: float, deadline_s: float, n_2d: int) -> bool:
    """A new vote may start: the stop point for new units, plus time for the 2d checks after the stage."""
    reserve = -(-n_2d // max(1, LLM_WORKERS)) * config.VERIFY_UNIT_S
    return time.monotonic() - t0 < deadline_s - config.stop_new_before_deadline_s() - reserve


def _confirm_units(answered: dict, group_of: dict, client, ledger, t0, deadline_s, settle, status, traj,
                   reserve: float, n_2d: int) -> list:
    """One unit per 2a group, best-paid groups first. A group of more than CONFIRM_GROUP_SAMPLE answered members is
    accepted when the FAST votes on its best-paid members all agree; otherwise every member is checked. Each member's
    calls are recorded on its own trajectory line. A math group (HARNESS_CONFIRM_MATH) takes MODEL_REASON's vote in
    place of the first FAST vote (`confirm_reason`) and runs after every group of CONFIRM_TYPES."""
    from harness.confirm import confirm, confirm_reason
    by_group: dict = {}
    for tid, (task, answer) in answered.items():
        if task.task_type not in config.CONFIRM_TYPES and not _reason_votes(task):
            continue
        by_group.setdefault(group_of.get(tid, tid), []).append((task, answer))

    def affordable(est: float) -> bool:
        return ledger.can_afford(est + reserve) and ledger.spent < config.CONFIRM_SPEND_LIMIT * ledger.budget

    def unit(members):
        members = sorted(members, key=lambda m: (-m[0].reward, m[0].task_id))
        # a REASON vote needs its whole call on the ledger (see _reason_fits); the third vote is a FAST program
        by_reason = _reason_votes(members[0][0])
        extra = REASON_CALL_TOKENS if by_reason else 0
        est = estimate_tokens(members[0][0]) + extra

        async def go():
            sample = config.CONFIRM_GROUP_SAMPLE if len(members) > config.CONFIRM_GROUP_SAMPLE else len(members)
            all_agree = True
            for i, (task, answer) in enumerate(members):
                if i >= sample and all_agree:
                    return   # the spec held on the sampled members
                vote_est = estimate_tokens(task)
                if not _confirm_time_left(t0, deadline_s, n_2d) or not affordable(vote_est + extra):
                    return
                start = time.monotonic()
                more = lambda: affordable(vote_est) and _confirm_time_left(t0, deadline_s, n_2d)  # noqa: E731
                if by_reason:
                    verdict = await confirm_reason(task, answer, client, solve_with_reason, confirm_program, more)
                else:
                    verdict = await confirm(task, answer, client, confirm_program, more)
                all_agree = all_agree and verdict.fast_agrees
                if verdict.replacement is not None:
                    settle(2, task, SolveResult("solved", answer=verdict.replacement, note=verdict.note),
                           time.monotonic() - start)
                    if status.get(task.task_id) != "solved" and (task.dir / "answer.txt").exists():
                        status[task.task_id] = "solved"   # the write failed: 2a's answer is still on disk
                else:
                    traj.record({"phase": 2, "task_id": task.task_id, "status": "solved", **_take_calls(),
                                 "elapsed": round(time.monotonic() - start, 3), "note": verdict.note[:200]})
        return {"prio": sum(t.reward for t, _ in members), "estimate": est + reserve, "run": go,
                "name": f"confirm:{members[0][0].task_id}", "by_reason": by_reason}

    units = [unit(m) for m in by_group.values()]
    # the data votes first, as in round 9 (the only measured fixes); the math REASON votes after them
    units.sort(key=lambda u: (u["by_reason"], -u["prio"]))
    return units
